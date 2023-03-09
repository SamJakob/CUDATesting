# Ensure correct generator and toolchain is set for Windows.
if (WIN32)
    if (NOT "${CMAKE_GENERATOR}" MATCHES "^Visual Studio")
        message(FATAL_ERROR "[Preload] To compile CUDA applications on Windows, you must use the Visual Studio generator (-G Visual Studio 17 2022). You are using ${CMAKE_GENERATOR}.")
    endif()
endif()
