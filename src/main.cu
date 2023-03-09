#include "benchmark.h"
#include "kernels/kernels.h"

#define SELECTED_KERNEL KERNEL_HISTOGRAMS

int main() {

#if SELECTED_KERNEL == KERNEL_ARRAY
    return boot_array();
#elif SELECTED_KERNEL == KERNEL_HISTOGRAMS
    return boot_histograms();
#else
    // No kernel selected.
    fprintf(stderr, "No kernel selected.\n");
    return 207;
#endif

}
