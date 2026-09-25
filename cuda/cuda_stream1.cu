#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// macro ตรวจ error ทุก call — ระหว่างฝึกควรใช้เสมอ จะได้รู้ทันทีว่าพังตรงไหน
#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t _e = (call);                                 \
        if (_e != cudaSuccess)                                   \
        {                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// kernel แบบ element-wise ที่จงใจใส่งานคำนวณเยอะ เพื่อให้ compute time
// ใกล้เคียงกับ transfer time — ถ้า kernel เบาไป overlap จะไม่เห็นผล
__global__ void heavy(const float *in, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        float x = in[i];
        for (int k = 0; k < 20; ++k)
            x = sinf(x) + sqrtf(fabsf(x)) * 0.5f;
        out[i] = x;
    }
}
int main()
{
    const int N = 1 << 24;

    const int nStreams = 4;
    const int chunkSize = N / nStreams; // สมมติ N หารลงตัวก่อน
    const size_t chunkBytes = (size_t)chunkSize * sizeof(float);
    const size_t bytes = (size_t)N * sizeof(float);

    // TODO 1: สร้าง array ของ stream: cudaStream_t stream[nStreams];
    //         แล้ว loop สร้างด้วย cudaStreamCreate
    cudaEvent_t start, stop;             // ประกาศตัวแปร event 2 ตัว
    CUDA_CHECK(cudaEventCreate(&start)); // จอง/สร้างจริง
    CUDA_CHECK(cudaEventCreate(&stop));

    cudaStream_t stream[nStreams];

    for (int c = 0; c < nStreams; ++c)
        CUDA_CHECK(cudaStreamCreate(&stream[c]));

    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)(i % 1000) * 0.001f;

    // --- device memory ---
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    const int threads = 256;
    // const int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(start));
    for (int c = 0; c < nStreams; ++c)
    {
        int off = c * chunkSize; // index เริ่มต้นของ chunk นี้

        CUDA_CHECK(cudaMemcpyAsync(d_in + off, h_in + off, chunkBytes,
                                   cudaMemcpyHostToDevice, stream[c]));

        int chunkBlocks = (chunkSize + threads - 1) / threads;
        heavy<<<chunkBlocks, threads, 0, stream[c]>>>(d_in + off, d_out + off, chunkSize);

        CUDA_CHECK(cudaMemcpyAsync(h_out + off, d_out + off, chunkBytes,
                                   cudaMemcpyDeviceToHost, stream[c]));
    }

    // TODO 5: loop cudaStreamDestroy ทุกตัว
    for (int c = 0; c < nStreams; ++c)
        CUDA_CHECK(cudaStreamDestroy(stream[c]));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop)); // รอให้ทุก stream เสร็จ

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("baseline (streams): %.2f ms\n", ms);
    printf("check: h_out[0]=%.4f  h_out[N-1]=%.4f\n", h_out[0], h_out[N - 1]);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("%d\n", prop.asyncEngineCount);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}