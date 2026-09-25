// ============================================================================
// sw_batch (BWA-MEM ksw_extend semantics) — ไฟล์เต็ม พร้อม compile/รัน
//
// signature ของ sw_batch / sw_multi_gpu คงเดิมทุกตัว -> caller เดิมไม่ต้องแก้
//
// เปลี่ยนจากของเดิม:
//   - kernel เป็น ksw_extend2 semantics (extension จาก h0, adaptive band, z-drop)
//   - DP ใช้ rolling row {h,e} ต่อ thread -> ตัด d_H/d_E/d_F/d_tb + memset ทิ้ง
//   - ตัด traceback/CIGAR (bwa สร้างทีหลังด้วย ksw_global คนละ phase)
//   - n_gpu อ่านจาก env N_GPU (default 2) เพื่อรัน 1/2/4 ได้โดยไม่ต้อง recompile
//
// ⚠ sequence ต้อง 2-bit encoded (A=0 C=1 G=2 T=3 N=4) ไม่ใช่ ASCII
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <pthread.h>
#include <cuda_runtime.h>
#include "bwa.h"

// ---- scoring: BWA-MEM defaults --------------------------------------------
#define MATCH 1
#define MISMATCH -4
#define AMBIG -1 // เจอ N (base 4)
#define O_DEL 6
#define E_DEL 1
#define O_INS 6
#define E_INS 1
#define END_BONUS 5
#define ZDROP 100

#define W_DEFAULT 100 // opt->w ของ bwa
#define H0_DEFAULT 20  // seed score (ใส่ของจริงเมื่อ trace มี h0)

#define DMAX(a, b) ((a) > (b) ? (a) : (b))
#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define IABS(x) ((x) < 0 ? -(x) : (x))

#define COUNT_CELLS

#define CUDA_CHECK(call)                                                  \
    do                                                                    \
    {                                                                     \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess)                                            \
        {                                                                 \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                              \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// ============================================================================
//                                 KERNEL
// ============================================================================
static inline char enc(char c) {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default:  return (c >= 0 && c <= 4) ? c : 4;
    }
}

__global__ void sw_batch_kernel(
    const char *reads, const int *q_off, const int *q_len,
    const char *refs, const int *r_off, const int *r_len,
    int n_tasks,
    int *out_score, int *out_qbeg, int *out_qend,
    int *out_rbeg, int *out_rend, char *out_cigar,
    const int *h0_arr, const int *w_arr)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_tasks)
        return;

    const unsigned char *query = (const unsigned char *)(reads + q_off[tid]);
    const unsigned char *target = (const unsigned char *)(refs + r_off[tid]);
    int qlen = q_len[tid];
    int tlen = r_len[tid];
    int h0 = h0_arr ? h0_arr[tid] : H0_DEFAULT;
    int w = w_arr ? w_arr[tid] : W_DEFAULT;

    if (qlen <= 0 || tlen <= 0 || qlen > MAX_QLEN)
    {
        out_score[tid] = -1;
        out_qbeg[tid] = 0;
        out_qend[tid] = 0;
        out_rbeg[tid] = 0;
        out_rend[tid] = 0;
        out_cigar[(size_t)tid * MAX_CIGAR] = '\0';
        return;
    }

    // rolling row: eh_h = H(i-1,*) , eh_e = E(i,*)
    int eh_h[MAX_QLEN + 1];
    int eh_e[MAX_QLEN + 1];

    const int oe_del = O_DEL + E_DEL;
    const int oe_ins = O_INS + E_INS;

    // หด w ถ้ากว้างเกินกว่าคะแนนจะไปถึง (เหมือน ksw_extend2)
    {
        int max_ins = (int)((double)(qlen * MATCH + END_BONUS - O_INS) / E_INS + 1.0);
        max_ins = max_ins > 1 ? max_ins : 1;
        w = w < max_ins ? w : max_ins;
        int max_del = (int)((double)(qlen * MATCH + END_BONUS - O_DEL) / E_DEL + 1.0);
        max_del = max_del > 1 ? max_del : 1;
        w = w < max_del ? w : max_del;
    }

    // ---- แถวแรก: ต่อจาก seed ด้วย h0 ----
    for (int j = 0; j <= qlen; ++j)
    {
        eh_h[j] = 0;
        eh_e[j] = 0;
    }
    eh_h[0] = h0;
    eh_h[1] = h0 > oe_ins ? h0 - oe_ins : 0;
    for (int j = 2; j <= qlen && eh_h[j - 1] > E_INS; ++j)
        eh_h[j] = eh_h[j - 1] - E_INS;

    int best = h0, max_i = -1, max_j = -1, max_ie = -1, gscore = -1, max_off = 0;
    int beg = 0, end = qlen;

#ifdef COUNT_CELLS
    unsigned int cells = 0;
#endif

    for (int i = 0; i < tlen; ++i)
    {
        int t, f = 0, h1, m = 0, mj = -1;
        unsigned char tc = target[i];

        if (beg < i - w)
            beg = i - w;
        if (end > i + w + 1)
            end = i + w + 1;
        if (end > qlen)
            end = qlen;

#ifdef COUNT_CELLS
        cells += (unsigned int)(end > beg ? end - beg : 0);
#endif

        if (beg == 0)
        {
            h1 = h0 - (O_DEL + E_DEL * (i + 1));
            if (h1 < 0)
                h1 = 0;
        }
        else
            h1 = 0;

        for (int j = beg; j < end; ++j)
        {
            int M = eh_h[j];        // H(i-1,j-1)
            int e = eh_e[j];        // E(i,j)
            eh_h[j] = h1;

            unsigned char qc = query[j];
            int sub = (qc > 3 || tc > 3) ? AMBIG : ((qc == tc) ? MATCH : MISMATCH);

            M = M ? M + sub : 0;    // ★ หัวใจ: cell ตายแล้วปลุกไม่ได้

            int h = M > e ? M : e;
            h = h > f ? h : f;
            h1 = h;

            mj = m > h ? mj : j;
            m = m > h ? m : h;

            t = M - oe_del;         // ★ ใช้ M
            t = t > 0 ? t : 0;
            e -= E_DEL;
            e = e > t ? e : t;
            eh_e[j] = e;

            t = M - oe_ins;         // ★ ใช้ M
            t = t > 0 ? t : 0;
            f -= E_INS;
            f = f > t ? f : t;
        }
        eh_h[end] = h1;
        eh_e[end] = 0;

        if (end == qlen)
        {
            max_ie = gscore > h1 ? max_ie : i;
            gscore = gscore > h1 ? gscore : h1;
        }

        if (m == 0)
            break;

        if (m > best)
        {
            best = m;
            max_i = i;
            max_j = mj;
            int off = IABS(mj - i);
            max_off = max_off > off ? max_off : off;
        }
        else if (ZDROP > 0)
        {
            if (i - max_i > mj - max_j)
            {
                if (best - m - ((i - max_i) - (mj - max_j)) * E_DEL > ZDROP)
                    break;
            }
            else
            {
                if (best - m - ((mj - max_j) - (i - max_i)) * E_INS > ZDROP)
                    break;
            }
        }
        // adaptive band narrowing
        {
            int j;
            for (j = beg; j < end && eh_h[j] == 0 && eh_e[j] == 0; ++j)
                ;
            beg = j;
            for (j = end; j >= beg && eh_h[j] == 0 && eh_e[j] == 0; --j)
                ;
            end = (j + 2 < qlen) ? j + 2 : qlen;
        }
              
    }

    out_score[tid] = best;
    out_qbeg[tid] = 0;         // extension เริ่มจากปลาย seed
    out_qend[tid] = max_j + 1; // qle
    out_rbeg[tid] = 0;
    out_rend[tid] = max_i + 1; // tle
    out_cigar[(size_t)tid * MAX_CIGAR] = '\0';

    (void)gscore;
    (void)max_ie;
    (void)max_off; // ต้องเพิ่ม out array ก่อนถึงจะเก็บได้
#ifdef COUNT_CELLS
    out_qbeg[tid] = (int)cells; 
#endif
}

// ============================================================================
//                             HOST: one GPU
// ============================================================================
void sw_batch(const char *h_reads, const int *h_qoff, const int *h_qlen, int qtot,
              const char *h_refs, const int *h_roff, const int *h_rlen, int rtot,
              int n_tasks,
              int *score, int *qbeg, int *qend, int *rbeg, int *rend, char *cigar)
{
    if (n_tasks <= 0)
        return;

    int CHUNK = 50000; // ไม่มี DP matrix ใน global แล้ว -> chunk ใหญ่ได้
    const char *env_chunk = getenv("SW_CHUNK");
    if (env_chunk)
        CHUNK = atoi(env_chunk);
    if (CHUNK <= 0)
        CHUNK = 200000;
    if (CHUNK > n_tasks)
        CHUNK = n_tasks;

    StreamBuf buf[N_STREAM];

    for (int s = 0; s < N_STREAM; s++)
    {
        StreamBuf *b = &buf[s];
        CUDA_CHECK(cudaStreamCreate(&b->stream));

        CUDA_CHECK(cudaMalloc((void **)&b->d_reads, (size_t)CHUNK * MAX_QLEN));
        CUDA_CHECK(cudaMalloc((void **)&b->d_refs, (size_t)CHUNK * MAX_RLEN));
        CUDA_CHECK(cudaMalloc((void **)&b->d_qoff, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_qlen, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_roff, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_rlen, CHUNK * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&b->d_score, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_qbeg, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_qend, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_rbeg, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_rend, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void **)&b->d_cigar, (size_t)CHUNK * MAX_CIGAR));

        CUDA_CHECK(cudaMallocHost((void **)&b->h_reads, (size_t)CHUNK * MAX_QLEN));
        CUDA_CHECK(cudaMallocHost((void **)&b->h_refs, (size_t)CHUNK * MAX_RLEN));
        CUDA_CHECK(cudaMallocHost((void **)&b->h_qoff, CHUNK * sizeof(int)));
        CUDA_CHECK(cudaMallocHost((void **)&b->h_roff, CHUNK * sizeof(int)));
    }

    for (int start = 0; start < n_tasks; start += CHUNK)
    {
        int s = (start / CHUNK) % N_STREAM;
        StreamBuf *b = &buf[s];
        CUDA_CHECK(cudaStreamSynchronize(b->stream));

        int n = (start + CHUNK <= n_tasks) ? CHUNK : (n_tasks - start);

        int qcur = 0, rcur = 0;
        for (int t = 0; t < n; ++t)
        {
            int g = start + t;
            b->h_qoff[t] = qcur;
            memcpy(b->h_reads + qcur, h_reads + h_qoff[g], h_qlen[g]);
            qcur += h_qlen[g];
            b->h_roff[t] = rcur;
            memcpy(b->h_refs + rcur, h_refs + h_roff[g], h_rlen[g]);
            rcur += h_rlen[g];
        }

        CUDA_CHECK(cudaMemcpyAsync(b->d_reads, b->h_reads, qcur, cudaMemcpyHostToDevice, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_refs, b->h_refs, rcur, cudaMemcpyHostToDevice, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_qoff, b->h_qoff, n * sizeof(int), cudaMemcpyHostToDevice, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_roff, b->h_roff, n * sizeof(int), cudaMemcpyHostToDevice, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_qlen, h_qlen + start, n * sizeof(int), cudaMemcpyHostToDevice, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(b->d_rlen, h_rlen + start, n * sizeof(int), cudaMemcpyHostToDevice, b->stream));
        

        int block = 128;
        int grid = (n + block - 1) / block;
        sw_batch_kernel<<<grid, block, 0, b->stream>>>(
            b->d_reads, b->d_qoff, b->d_qlen,
            b->d_refs, b->d_roff, b->d_rlen,
            n,
            b->d_score, b->d_qbeg, b->d_qend, b->d_rbeg, b->d_rend, b->d_cigar,
            nullptr, nullptr); // h0_arr, w_arr — ใส่ของจริงเมื่อ trace พร้อม
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(score + start, b->d_score, n * sizeof(int), cudaMemcpyDeviceToHost, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(qbeg + start, b->d_qbeg, n * sizeof(int), cudaMemcpyDeviceToHost, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(qend + start, b->d_qend, n * sizeof(int), cudaMemcpyDeviceToHost, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(rbeg + start, b->d_rbeg, n * sizeof(int), cudaMemcpyDeviceToHost, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(rend + start, b->d_rend, n * sizeof(int), cudaMemcpyDeviceToHost, b->stream));
        CUDA_CHECK(cudaMemcpyAsync(cigar + (size_t)start * MAX_CIGAR, b->d_cigar,
                                   (size_t)n * MAX_CIGAR, cudaMemcpyDeviceToHost, b->stream));
    }

    for (int s = 0; s < N_STREAM; s++)
    {
        StreamBuf *b = &buf[s];
        CUDA_CHECK(cudaStreamSynchronize(b->stream));
        cudaStreamDestroy(b->stream);
        cudaFree(b->d_reads);
        cudaFree(b->d_refs);
        cudaFree(b->d_qoff);
        cudaFree(b->d_qlen);
        cudaFree(b->d_roff);
        cudaFree(b->d_rlen);
        cudaFree(b->d_score);
        cudaFree(b->d_qbeg);
        cudaFree(b->d_qend);
        cudaFree(b->d_rbeg);
        cudaFree(b->d_rend);
        cudaFree(b->d_cigar);
        cudaFreeHost(b->h_reads);
        cudaFreeHost(b->h_refs);
        cudaFreeHost(b->h_qoff);
        cudaFreeHost(b->h_roff);
    }
}

// ============================================================================
//                            HOST: multi GPU
// ============================================================================
typedef struct
{
    int dev, start, n;
    const char *h_reads, *h_refs;
    const int *h_qoff, *h_qlen, *h_roff, *h_rlen;
    int *score, *qbeg, *qend, *rbeg, *rend;
    char *cigar;
} GpuJob;

static void *gpu_worker(void *arg)
{
    GpuJob *j = (GpuJob *)arg;
    if (j->n <= 0)
        return NULL;
    cudaError_t e = cudaSetDevice(j->dev);
    if (e != cudaSuccess)
    {
        fprintf(stderr, "GPU %d: cudaSetDevice failed: %s\n", j->dev, cudaGetErrorString(e));
        return NULL;
    }
    sw_batch(j->h_reads, j->h_qoff + j->start, j->h_qlen + j->start, 0,
             j->h_refs, j->h_roff + j->start, j->h_rlen + j->start, 0,
             j->n,
             j->score + j->start, j->qbeg + j->start, j->qend + j->start,
             j->rbeg + j->start, j->rend + j->start,
             j->cigar + (size_t)j->start * MAX_CIGAR);
    return NULL;
}

extern "C" void sw_multi_gpu(const char *h_reads, const int *h_qoff, const int *h_qlen, int qtot,
                             const char *h_refs, const int *h_roff, const int *h_rlen, int rtot,
                             int n_tasks,
                             int *score, int *qbeg, int *qend, int *rbeg, int *rend, char *cigar)
{
    int avail = 0;
    cudaError_t e = cudaGetDeviceCount(&avail);
    if (e != cudaSuccess || avail == 0)
    {
        fprintf(stderr, "no CUDA device: %s\n", cudaGetErrorString(e));
        return;
    }

    int n_gpu = 2;
    const char *env_gpu = getenv("N_GPU"); // รัน 1/2/4 ได้โดยไม่ต้อง recompile
    if (env_gpu)
        n_gpu = atoi(env_gpu);
    if (n_gpu <= 0)
        n_gpu = 1;
    if (n_gpu > avail)
    {
        fprintf(stderr, "want %d GPU but only %d available -> use %d\n", n_gpu, avail, avail);
        n_gpu = avail;
    }
    if (n_gpu > 8)
        n_gpu = 8;
    printf("using %d / %d GPU\n", n_gpu, avail);

    int per = (n_tasks + n_gpu - 1) / n_gpu;
    pthread_t th[8];
    GpuJob job[8];

    for (int d = 0; d < n_gpu; d++)
    {
        job[d].dev = d;
        job[d].start = d * per;
        job[d].n = (job[d].start + per <= n_tasks) ? per : (n_tasks - job[d].start);
        if (job[d].n < 0)
            job[d].n = 0;
        job[d].h_reads = h_reads;
        job[d].h_refs = h_refs;
        job[d].h_qoff = h_qoff;
        job[d].h_qlen = h_qlen;
        job[d].h_roff = h_roff;
        job[d].h_rlen = h_rlen;
        job[d].score = score;
        job[d].qbeg = qbeg;
        job[d].qend = qend;
        job[d].rbeg = rbeg;
        job[d].rend = rend;
        job[d].cigar = cigar;
        pthread_create(&th[d], NULL, gpu_worker, &job[d]);
    }
    for (int d = 0; d < n_gpu; d++)
        pthread_join(th[d], NULL);
}
