#ifndef LIB_SAMUDA
#define LIB_SAMUDA true

/* 'Extreme' SAMUDA lib. Includes lots of wacky customized features that are likely
 * unintuitive to ordinary C++ or CUDA developers.
 */

#ifdef LIB_SAMUDA_ANGELEAKY

    /* Ordinary C++ aliases */
    #define var auto
    #define uint unsigned int

    /* Aliases for device or host prefixes for variables. */

    // A variable that references memory on the host.
    #define on_host(x) h_##x

    // A variable that references memory on the device.
    #define on_device(x) d_##x

#endif

/* Aliases for position calculations. */

#define block_position(dim) blockDim.dim * blockIdx.dim + threadIdx.dim;
#define thread_position(dim) threadIdx.dim;

#endif
