#include <algorithm>
#include <atomic>
#include <iostream>
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <numbers>
#include <fftw3.h>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

int width = 800;
int height = 600;

constexpr int fftWindow = 1024;
constexpr int fftBin = (fftWindow / 2) + 1;

constexpr int nBars = 64;
constexpr int nBinsPerBar = fftBin / nBars;

constexpr float MIN_DB = -60.0f;
constexpr float MAX_DB = 0.0f;
constexpr float DB_RANGE = MAX_DB - MIN_DB;

float audioBuffer[fftWindow] = {0};
double renderBuffer[fftWindow] = {0};

std::atomic<size_t> filledSize{0};

void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount)
{
    (void)pInput;

    ma_decoder* pDecoder = (ma_decoder*)pDevice->pUserData;
    if (pDecoder == NULL) {
        return;
    }

    ma_decoder_read_pcm_frames(pDecoder, pOutput, frameCount, NULL);

    float* pOut = (float*)(pOutput);
    size_t nChannels = pDecoder->outputChannels;

    size_t writeIdx = filledSize.load(std::memory_order_relaxed);
    for(size_t i = 0; i < frameCount; ++i)
    {
        float mono = 0;
        for(size_t j = 0; j < nChannels; j++)
        {
            mono += pOut[i * nChannels + j];
        }
        mono /= nChannels;

        audioBuffer[writeIdx] = mono;
        writeIdx = (writeIdx + 1) % fftWindow;
    }
    filledSize.store(writeIdx ,std::memory_order_release);
}

int main(int argc, char* argv[])
{
    bool error = false;

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Event event;

    ma_result decoderResult = MA_DEVICE_NOT_INITIALIZED;
    ma_result deviceResult = MA_DEVICE_NOT_INITIALIZED;
    ma_decoder decoder;
    ma_device_config deviceConfig;
    ma_device device;
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 0, 0);

    std::vector<double> hannTable(fftWindow);
    std::vector<float> barHeights(nBars, 0);

    for(size_t i = 0; i < fftWindow; i++) {
        hannTable[i] = 0.5f * (1.0f - std::cos((2.0f * std::numbers::pi_v<double> * i) / (fftWindow - 1)));
    }

    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " <audio_path>\n";
        return 1;
    }

    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftBin);
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

    if (ma_device_start(&device) != MA_SUCCESS) {
        std::cerr << "Failed to start playback device.\n";
        error = true;
        goto cleanup;
    }

    while(1)
    {
        SDL_PollEvent(&event);
        if(event.type == SDL_EVENT_QUIT)
        {
            break;
        }
        else if(event.type == SDL_EVENT_WINDOW_RESIZED)
        {
            SDL_GetWindowSize(window, &width, &height);
        }
        else if(event.type == SDL_EVENT_KEY_DOWN)
        {
            if(event.key.scancode == SDL_SCANCODE_ESCAPE)
                break;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
        SDL_RenderClear(renderer);

        {
            size_t currentIdx = filledSize.load(std::memory_order_acquire);
            for(size_t i = 0; i < fftWindow; i++) {
                size_t readIdx = (currentIdx + i) % fftWindow;
                renderBuffer[i] = (double)audioBuffer[readIdx] * hannTable[i];
            }

            fftw_execute(plan);

            float barWidth = (float)width / nBars;
            for(int i = 0; i < nBars; i++) {
                double mag = 0;
                for(int j = 0; j < nBinsPerBar; j++) {
                    double r = out[i * nBinsPerBar + j][0];
                    double img = out[i * nBinsPerBar + j][1];
                    mag += std::sqrt(r*r + img*img);
                }
                mag /= nBinsPerBar;

                float dbValue = 20.0f * std::log10(mag + 1e-6f);

                float normalized = (dbValue - MIN_DB) / DB_RANGE;
                normalized = std::clamp(normalized, 0.0f, 1.0f);
                normalized *= normalized;

                float targetHeight = normalized * (height / 2.0f) * 0.8f;
                barHeights[i] += (targetHeight - barHeights[i]) * 0.2f;
                float renderHeight = barHeights[i];

                float startY = (height / 2.0f) - renderHeight;

                SDL_FRect bar = {
                    (float)i * barWidth,
                    startY,
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
