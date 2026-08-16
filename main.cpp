#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <numbers>
#include <fftw3.h>
#include <vector>
#include <array>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

int width = 800;
int height = 600;

constexpr size_t fftWindow = 1024;
constexpr size_t fftBin = (fftWindow / 2) + 1;

constexpr size_t nBars = 64;
constexpr float MIN_FREQ = 30.0f;

constexpr float MIN_DB = -60.0f;
constexpr float MAX_DB = 0.0f;
constexpr float DB_RANGE = MAX_DB - MIN_DB;

// How quickly a bar chases its target, as a time constant in seconds.
// Expressed this way the decay looks the same at 60Hz and at 144Hz.
constexpr float SMOOTH_TAU = 0.06f;

// Touched only by the audio thread.
float audioRing[fftWindow] = {0};
size_t ringWrite = 0;

// Touched only by the render thread.
double renderBuffer[fftWindow] = {0};

// The handoff between the two. The audio thread publishes the most recent
// fftWindow samples under a seqlock; an odd sequence number means a write is
// in flight, so the reader retries. Elements are atomic so that a read racing
// the writer is merely stale rather than undefined behaviour.
std::atomic<float> snapshot[fftWindow] = {};
std::atomic<uint32_t> snapshotSeq{0};

std::atomic<bool> playbackFinished{false};

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    (void)pInput;

    ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
    if (pDecoder == NULL) {
        return;
    }

    ma_uint64 framesRead = 0;
    ma_decoder_read_pcm_frames(pDecoder, pOutput, frameCount, &framesRead);

    float* pOut = (float*)(pOutput);
    size_t nChannels = pDecoder->outputChannels;

    // A short read means the file ran out. Silence whatever the decoder left
    // untouched, otherwise the tail of the last buffer plays as garbage.
    if(framesRead < frameCount)
    {
        std::fill(pOut + framesRead * nChannels, pOut + (size_t)frameCount * nChannels, 0.0f);
        playbackFinished.store(true, std::memory_order_relaxed);
    }

    for(size_t i = 0; i < frameCount; ++i)
    {
        float mono = 0;
        for(size_t j = 0; j < nChannels; j++)
        {
            mono += pOut[i * nChannels + j];
        }
        mono /= nChannels;

        audioRing[ringWrite] = mono;
        ringWrite = (ringWrite + 1) % fftWindow;
    }

    // Publish the ring, oldest sample first, so the reader gets a coherent
    // window instead of one being overwritten underneath it.
    uint32_t seq = snapshotSeq.load(std::memory_order_relaxed);
    snapshotSeq.store(seq + 1, std::memory_order_release);
    for(size_t i = 0; i < fftWindow; i++)
    {
        snapshot[i].store(audioRing[(ringWrite + i) % fftWindow], std::memory_order_relaxed);
    }
    snapshotSeq.store(seq + 2, std::memory_order_release);
}

int main(int argc, char* argv[])
{
    bool error = false;
    bool running = true;

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Event event{};

    uint64_t prevTicks = 0;

    ma_result decoderResult = MA_DEVICE_NOT_INITIALIZED;
    ma_result deviceResult = MA_DEVICE_NOT_INITIALIZED;
    ma_decoder decoder;
    ma_device_config deviceConfig;
    ma_device device;
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 0, 0);

    std::vector<double> hannTable(fftWindow);
    std::vector<float> barHeights(nBars, 0);
    std::array<size_t, nBars + 1> bandEdges{};

    for(size_t i = 0; i < fftWindow; i++) {
        hannTable[i] = 0.5f * (1.0f - std::cos((2.0f * std::numbers::pi_v<double> * i) / (fftWindow - 1)));
    }

    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <audio_path>\n";
        return 1;
    }

    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftBin);
    if(out == nullptr)
    {
        std::cerr << "Failed to allocate FFT output buffer.\n";
        return 1;
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(fftWindow, renderBuffer, out, FFTW_ESTIMATE);

    //miniaudio init
    decoderResult = ma_decoder_init_file(/*audio_path*/ argv[1], &decoderConfig, &decoder);
    if(decoderResult != MA_SUCCESS)
    {
        std::cerr << "Failed to initialize decoder.\n";
        error = true;
        goto cleanup;
    }

    deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = decoder.outputFormat;
    deviceConfig.playback.channels = decoder.outputChannels;
    deviceConfig.sampleRate        = decoder.outputSampleRate;
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &decoder;

    deviceResult = ma_device_init(nullptr, &deviceConfig, &device);
    if (deviceResult != MA_SUCCESS) {
        std::cerr << "Failed to open playback device.\n";
        error = true;
        goto cleanup;
    }

    //sdl init
    if(!SDL_Init(SDL_INIT_VIDEO))
    {
        SDL_Log("Couldn't initialize SDL: %s", SDL_GetError());
        error = true;
        goto cleanup;
    }

    if(!SDL_CreateWindowAndRenderer("Visualizer", width, height, SDL_WINDOW_RESIZABLE, &window, &renderer))
    {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't create window and renderer: %s", SDL_GetError());
        error = true;
        goto cleanup;
    }

    // Without this the loop spins as fast as the CPU allows, re-running the
    // FFT thousands of times a second for no visible benefit.
    if(!SDL_SetRenderVSync(renderer, 1))
        SDL_Log("Couldn't enable vsync, falling back to an unthrottled loop: %s", SDL_GetError());

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start playback device.\n";
        error = true;
        goto cleanup;
    }

    // Log-spaced band edges, so every bar covers the same musical interval.
    // Linear spacing would cram all the music into the first few bars.
    {
        double sampleRate = decoder.outputSampleRate;
        double ratio = (sampleRate / 2.0) / MIN_FREQ;

        for(size_t i = 0; i <= nBars; i++) {
            double freq = MIN_FREQ * std::pow(ratio, (double)i / nBars);
            size_t bin = (size_t)std::lround(freq * fftWindow / sampleRate);

            // the lowest bands all round to the same bin, so force them apart
            if(i > 0 && bin <= bandEdges[i - 1])
                bin = bandEdges[i - 1] + 1;

            bandEdges[i] = std::min(bin, fftBin - 1);
        }
    }

    prevTicks = SDL_GetTicksNS();

    while(running)
    {
        // Drain the queue. Polling once per frame leaves events to pile up,
        // and leaves `event` holding a stale value when the queue is empty.
        while(SDL_PollEvent(&event))
        {
            if(event.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
            else if(event.type == SDL_EVENT_WINDOW_RESIZED)
            {
                SDL_GetWindowSize(window, &width, &height);
            }
            else if(event.type == SDL_EVENT_KEY_DOWN)
            {
                if(event.key.scancode == SDL_SCANCODE_ESCAPE)
                    running = false;
            }
        }

        if(playbackFinished.load(std::memory_order_relaxed))
            running = false;

        uint64_t nowTicks = SDL_GetTicksNS();
        float dt = (float)(nowTicks - prevTicks) / 1e9f;
        prevTicks = nowTicks;
        dt = std::clamp(dt, 0.0f, 0.1f); // don't let a stall snap every bar at once

        float smoothing = 1.0f - std::exp(-dt / SMOOTH_TAU);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderClear(renderer);

        {
            // Seqlock read: retry while the audio thread is mid-publish. The
            // cap keeps a preempted writer from stalling the frame; the worst
            // case is one visibly torn window.
            for(int attempt = 0; attempt < 8; attempt++)
            {
                uint32_t before = snapshotSeq.load(std::memory_order_acquire);
                if(before & 1u)
                    continue;

                for(size_t i = 0; i < fftWindow; i++)
                    renderBuffer[i] = (double)snapshot[i].load(std::memory_order_relaxed) * hannTable[i];

                if(snapshotSeq.load(std::memory_order_acquire) == before)
                    break;
            }

            fftw_execute(plan);

            // fftWindow/4 = fftWindow/2 for the transform, halved again for the
            // Hann window's coherent gain. A full-scale sine now lands at 0 dB.
            constexpr double magScale = fftWindow / 4.0;

            float barWidth = (float)width / nBars;
            for(size_t i = 0; i < nBars; i++) {
                // peak rather than average: the upper bands are dozens of bins
                // wide, and averaging them flattens every transient out
                double mag = 0;
                for(size_t j = bandEdges[i]; j < bandEdges[i + 1]; j++) {
                    double r = out[j][0];
                    double img = out[j][1];
                    mag = std::max(mag, std::sqrt(r*r + img*img) / magScale);
                }

                float dbValue = 20.0f * std::log10(mag + 1e-6f);

                float normalized = (dbValue - MIN_DB) / DB_RANGE;
                normalized = std::clamp(normalized, 0.0f, 1.0f);
                normalized *= normalized;

                float targetHeight = normalized * height * 0.9f;
                barHeights[i] += (targetHeight - barHeights[i]) * smoothing;
                float renderHeight = barHeights[i];

                SDL_FRect bar = {
                    (float)i * barWidth,
                    (float)height - renderHeight, // grow up from the bottom edge
                    std::max(1.0f, barWidth - 2.0f), // -2 for a small gap between bars
                    renderHeight
                };

                SDL_SetRenderDrawColor(renderer, 43, 201, 88, 255);
                SDL_RenderFillRect(renderer, &bar);
            }
        }

        SDL_RenderPresent(renderer);
    }

cleanup:
    fftw_destroy_plan(plan);
    fftw_free(out);

    if(deviceResult == MA_SUCCESS)
        ma_device_uninit(&device);
    if(decoderResult == MA_SUCCESS)
        ma_decoder_uninit(&decoder);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    if(error)
        return 1;

    return 0;
}
