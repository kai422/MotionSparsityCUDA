#ifndef COMMON
#define COMMON

#include <cuda.h>
#include <torch/extension.h>

#include <vector>

#define _unused(x) ((void)(x))

template <class T>
class ptr_wrapper {
 public:
  ptr_wrapper() : ptr(nullptr) {}
  ptr_wrapper(T *ptr) : ptr(ptr) {}
  ptr_wrapper(const ptr_wrapper &other) : ptr(other.ptr) {}
  T &operator*() const { return *ptr; }
  T *operator->() const { return ptr; }
  T *get() const { return ptr; }
  void destroy() { delete ptr; }
  T &operator[](std::size_t idx) const { return ptr[idx]; }

 private:
  T *ptr;
};

// check if there is a error after kernel execution
#define CUDA_POST_KERNEL_CHECK       \
  CUDA_CHECK(cudaPeekAtLastError()); \
  CUDA_CHECK(cudaGetLastError());

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

// Use 1024 threads per block, which requires cuda sm_2x or above
const int CUDA_NUM_THREADS = 1024;

// CUDA: number of blocks for threads.
inline int GET_BLOCKS_T(const int N, const int N_THREADS) {
  return (N + N_THREADS - 1) / N_THREADS;
}
inline int GET_BLOCKS(const int N) { return GET_BLOCKS_T(N, CUDA_NUM_THREADS); }

/// Allocate memory on the current GPU device.
/// @param N lenght of the array.
/// @return pointer to device array.
template <typename T>
T *device_malloc(long N) {
  T *dptr;
  CUDA_CHECK(cudaMalloc(&dptr, N * sizeof(T)));
  if (DEBUG) {
    printf("[DEBUG] device_malloc %p, %ld\n", dptr, N);
  }
  return dptr;
}

/// Frees device memory.
/// @param dptr
template <typename T>
void device_free(T *dptr) {
  if (DEBUG) {
    printf("[DEBUG] device_free %p\n", dptr);
  }
  CUDA_CHECK(cudaFree(dptr));
}

/// Copies host to device memory.
/// @param hptr
/// @param dptr
/// @param N
template <typename T>
void host_to_device(const T *hptr, T *dptr, long N) {
  if (DEBUG) {
    printf("[DEBUG] host_to_device %p => %p, %ld\n", hptr, dptr, N);
  }
  CUDA_CHECK(cudaMemcpy(dptr, hptr, N * sizeof(T), cudaMemcpyHostToDevice));
}

/// Copies host memory the newly allocated device memory.
/// @param hptr
/// @param N
/// @return dptr
template <typename T>
T *host_to_device_malloc(const T *hptr, long N) {
  T *dptr = device_malloc<T>(N);
  host_to_device(hptr, dptr, N);
  return dptr;
}

/// Copies device to host memory.
/// @param dptr
/// @param hptr
/// @param N
template <typename T>
void device_to_host(const T *dptr, T *hptr, long N) {
  if (DEBUG) {
    printf("[DEBUG] device_to_host %p => %p, %ld\n", dptr, hptr, N);
  }
  CUDA_CHECK(cudaMemcpy(hptr, dptr, N * sizeof(T), cudaMemcpyDeviceToHost));
}
/// Copies device memory the newly allocated host memory.
/// @param dptr
/// @param N
/// @return hptr
template <typename T>
T *device_to_host_malloc(const T *dptr, long N) {
  T *hptr = new T[N];
  device_to_host(dptr, hptr, N);
  return hptr;
}

/// Copies device to device memory.
/// @param src
/// @param dst
/// @param N
template <typename T>
void device_to_device(const T *src, T *dst, long N) {
  if (DEBUG) {
    printf("[DEBUG] device_to_device %p => %p, %ld\n", src, dst, N);
  }
  CUDA_CHECK(cudaMemcpy(dst, src, N * sizeof(T), cudaMemcpyDeviceToDevice));
}
#endif
