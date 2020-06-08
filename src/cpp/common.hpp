#ifndef COMMON
#define COMMON
#include <atomic>
#include <omp.h>

#include <torch/extension.h>
#ifndef _OPENMP
#define _OPENMP
#endif
#define _unused(x) ((void)(x))

template <class T>
class ptr_wrapper
{
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

inline int64_t divup(int64_t x, int64_t y)
{
    return (x + y - 1) / y;
}

template <class F>
inline void parallel_for(
    const int64_t begin,
    const int64_t end,
    const int64_t grain_size,
    const F &f)
{
    TORCH_CHECK(grain_size >= 0);
    if (begin >= end)
    {
        return;
    }
#ifdef _OPENMP
    std::atomic_flag err_flag = ATOMIC_FLAG_INIT;
    std::exception_ptr eptr;

#pragma omp parallel if (!omp_in_parallel() && ((end - begin) > grain_size))
    {
        // choose number of tasks based on grain size and number of threads
        // can't use num_threads clause due to bugs in GOMP's thread pool (See #32008)
        int64_t num_threads = omp_get_num_threads();
        if (grain_size > 0)
        {
            num_threads = std::min(num_threads, divup((end - begin), grain_size));
        }

        int64_t tid = omp_get_thread_num();
        int64_t chunk_size = divup((end - begin), num_threads);
        int64_t begin_tid = begin + tid * chunk_size;
        if (begin_tid < end)
        {
            try
            {
                f(begin_tid, std::min(end, chunk_size + begin_tid));
            }
            catch (...)
            {
                if (!err_flag.test_and_set())
                {
                    eptr = std::current_exception();
                }
            }
        }
    }
    if (eptr)
    {
        std::rethrow_exception(eptr);
    }
#else
    f(begin, end);
#endif
}

#endif
