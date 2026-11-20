// Read-only CUDA device limits for interpreting recorded launch geometry.
#include <cuda_runtime_api.h>
#include <iostream>
int main() {
    cudaDeviceProp p{};
    const auto error = cudaGetDeviceProperties(&p, 0);
    if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error); return 1; }
    int clock = 0, bus = 0;
    cudaDeviceGetAttribute(&clock, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    std::cout << "{\"name\":\"" << p.name << "\",\"sm_count\":" << p.multiProcessorCount
      << ",\"threads_per_sm\":" << p.maxThreadsPerMultiProcessor
      << ",\"registers_per_sm\":" << p.regsPerMultiprocessor
      << ",\"shared_bytes_per_sm\":" << p.sharedMemPerMultiprocessor
      << ",\"blocks_per_sm\":" << p.maxBlocksPerMultiProcessor
      << ",\"warp_size\":" << p.warpSize << ",\"l2_bytes\":" << p.l2CacheSize
      << ",\"total_memory_bytes\":" << p.totalGlobalMem
      << ",\"memory_clock_khz\":" << clock << ",\"memory_bus_bits\":" << bus
      << ",\"reported_peak_memory_bytes_per_second\":" << (2.0 * clock * 1000 * bus / 8)
      << ",\"note\":\"Device limits and nominal clock, not measured bandwidth or achieved occupancy\"}\n";
}
