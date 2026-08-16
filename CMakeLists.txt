cmake_minimum_required(VERSION 3.16)

project(music_visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Plain `cmake ..` leaves this empty, which means no optimization flags at all.
# Default to Release so the obvious invocation doesn't produce a debug build.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)

find_package(SDL3 CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)

add_executable(music_visualizer main.cpp)

target_link_libraries(music_visualizer
    PRIVATE
        SDL3::SDL3
        PkgConfig::FFTW
)

include(GNUInstallDirs)
install(TARGETS music_visualizer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
