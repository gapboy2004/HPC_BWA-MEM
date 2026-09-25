#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

// macro ตรวจ error ทุก call — ระหว่างฝึกควรใช้เสมอ จะได้รู้ทันทีว่าพังตรงไหน
#define CUDA_CHECK(call) do {                                   \
    cudaError_t _e = (call);                                    \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(_e));    \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while (0)

// kernel แบบ element-wise ที่จงใจใส่งานคำนวณเยอะ เพื่อให้ compute time
// ใกล้เคียงกับ transfer time — ถ้า kernel เบาไป overlap จะไม่เห็นผล
__global__ void heavy(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        for (int k = 0; k < 20; ++k)
            x = sinf(x) + sqrtf(fabsf(x)) * 0.5f;
        out[i] = x;
    }
}

int main() {
    const int    N     = 1 << 24;              // ~16.7M floats = 64 MB
    const size_t bytes = (size_t)N * sizeof(float);

    // --- host memory: ใช้ pinned (cudaMallocHost) ตั้งแต่ baseline ---
    // เพื่อให้ตัวแปรเดียวที่ต่างระหว่าง baseline กับ multi-stream คือ "การมี stream"
    float *h_in, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in,  bytes));
    CUDA_CHECK(cudaMallocHost(&h_out, bytes));
    for (int i = 0; i < N; ++i) h_in[i] = (float)(i % 1000) * 0.001f;

    // --- device memory ---
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    // จับเวลาด้วย CUDA event (วัดเวลา GPU จริง ไม่ใช่เวลา wall-clock ฝั่ง host)
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // ====== หัวใจของ baseline: ทุกอย่างทำเรียงกัน ไม่มี overlap ======
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice)); // (1) H2D ทั้งก้อน
    heavy<<<blocks, threads>>>(d_in, d_out, N);                        // (2) kernel ทั้งก้อน
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));// (3) D2H ทั้งก้อน
    // ================================================================

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("baseline (no streams): %.2f ms\n", ms);
    printf("check: h_out[0]=%.4f  h_out[N-1]=%.4f\n", h_out[0], h_out[N - 1]);

    // cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
