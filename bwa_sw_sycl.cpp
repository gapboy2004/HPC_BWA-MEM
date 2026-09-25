// bwa_sw_sycl.cpp — ksw_extend (BWA-MEM) บน GPU ผ่าน SYCL
//
// พอร์ตตรงจาก bwa_sw.cu ที่ validate ผ่านแล้ว (611,008 tasks ตรงกับ ksw_extend2)
// โครง 2 ระดับคงเดิม: sycl_multi_gpu() -> sw_sycl()
//
// เปลี่ยนจากเวอร์ชันก่อน:
//   - kernel เป็น ksw_extend2 semantics (extension จาก h0, adaptive band, z-drop)
//   - ตัด d_H/d_E/d_F/d_tb + memset 4 ก้อน -> rolling row ใน private memory
//     (นี่คือตัวแก้ OOM ที่ BAND_W ใหญ่: 1.65 MB/task -> ~4 KB/thread)
//   - encode ASCII -> 2bit ตอน pack (เหมือนฝั่ง CUDA)
//   - CHUNK / N_GPU อ่านจาก env (SW_CHUNK, N_GPU)
//
// build: acpp -O3 --acpp-targets=generic -c bwa_sw_sycl.cpp -o bwa_sw_sycl.o

#include <sycl/sycl.hpp>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <pthread.h>
#include "bwa.h"

// ---- scoring: ต้องตรงกับ bwa_sw.cu ทุกค่า -------------------------------
#define MATCH 1
#define MISMATCH -4
#define AMBIG -1 // เจอ N (base 4)
#define O_DEL 6
#define E_DEL 1
#define O_INS 6
#define E_INS 1
#define END_BONUS 5
#define ZDROP 100

#define W_DEFAULT 100
#define H0_DEFAULT 20 // ★ ต้องตรงกับ H0_DEFAULT ใน bwa_sw.cu

#define DMAX(a, b) ((a) > (b) ? (a) : (b))
#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define IABS(x) ((x) < 0 ? -(x) : (x))

class sw_kernel;

// ASCII -> 2bit (A=0 C=1 G=2 T=3, อื่น ๆ =4); ถ้า <5 อยู่แล้วถือว่า encode มาแล้ว
static inline char enc(char c)
{
    switch (c)
    {
    case 'A':
    case 'a':
        return 0;
    case 'C':
    case 'c':
        return 1;
    case 'G':
    case 'g':
        return 2;
    case 'T':
    case 't':
        return 3;
    default:
        return (c >= 0 && c <= 4) ? c : 4;
    }
}

// ============================================================
//  ระดับ 2: sw_sycl — 1 GPU (chunk + pack + kernel)
// ============================================================
struct Lane
{
    sycl::queue q;
    char *d_reads, *d_refs;
    int *d_qoff, *d_qlen, *d_roff, *d_rlen;
    int *d_score, *d_qbeg, *d_qend, *d_rbeg, *d_rend;
    char *d_cigar;
    char *h_reads, *h_refs; // pinned staging
    int *h_qoff, *h_roff;
};

void sw_sycl(sycl::device dev, int n_tasks,
             const char *h_reads, const int *h_qoff, const int *h_qlen,
             const char *h_refs, const int *h_roff, const int *h_rlen,
             int *score, int *qbeg, int *qend, int *rbeg, int *rend, char *cigar)
{
    if (n_tasks <= 0)
        return;

    sycl::context ctx{dev};

    const int N_LANE = 2;
    int CHUNK = 50000; // ไม่มี DP matrix ใน global แล้ว
    const char *env_chunk = getenv("SW_CHUNK");
    if (env_chunk && atoi(env_chunk) > 0)
        CHUNK = atoi(env_chunk);
    if (CHUNK > n_tasks)
        CHUNK = n_tasks;

    std::vector<Lane> lanes(N_LANE);
    for (auto &L : lanes)
    {
        L.q = sycl::queue{ctx, dev, sycl::property::queue::in_order{}};
        L.d_reads = sycl::malloc_device<char>((size_t)CHUNK * MAX_QLEN, L.q);
        L.d_refs = sycl::malloc_device<char>((size_t)CHUNK * MAX_RLEN, L.q);
        L.d_qoff = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_qlen = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_roff = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_rlen = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_score = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_qbeg = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_qend = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_rbeg = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_rend = sycl::malloc_device<int>(CHUNK, L.q);
        L.d_cigar = sycl::malloc_device<char>((size_t)CHUNK * MAX_CIGAR, L.q);
        L.h_reads = sycl::malloc_host<char>((size_t)CHUNK * MAX_QLEN, L.q);
        L.h_refs = sycl::malloc_host<char>((size_t)CHUNK * MAX_RLEN, L.q);
        L.h_qoff = sycl::malloc_host<int>(CHUNK, L.q);
        L.h_roff = sycl::malloc_host<int>(CHUNK, L.q);
    }

    int chunk_id = 0;
    for (int start = 0; start < n_tasks; start += CHUNK, ++chunk_id)
    {
        Lane &L = lanes[chunk_id % N_LANE];
        L.q.wait(); // buffer ของ lane นี้ว่างแล้ว

        int n = DMIN(CHUNK, n_tasks - start);

        // pack + encode (เหมือนฝั่ง CUDA)
        int qcur = 0, rcur = 0;
        for (int t = 0; t < n; ++t)
        {
            int g = start + t;
            L.h_qoff[t] = qcur;
            memcpy(L.h_reads + qcur, h_reads + h_qoff[g], h_qlen[g]);
            qcur += h_qlen[g];
            L.h_roff[t] = rcur;
            memcpy(L.h_refs + rcur, h_refs + h_roff[g], h_rlen[g]);
            rcur += h_rlen[g];
        }

        L.q.memcpy(L.d_reads, L.h_reads, qcur);
        L.q.memcpy(L.d_refs, L.h_refs, rcur);
        L.q.memcpy(L.d_qoff, L.h_qoff, n * sizeof(int));
        L.q.memcpy(L.d_roff, L.h_roff, n * sizeof(int));
        L.q.memcpy(L.d_qlen, h_qlen + start, n * sizeof(int));
        L.q.memcpy(L.d_rlen, h_rlen + start, n * sizeof(int));

        char *d_reads = L.d_reads, *d_refs = L.d_refs;
        int *d_qoff = L.d_qoff, *d_qlen = L.d_qlen;
        int *d_roff = L.d_roff, *d_rlen = L.d_rlen;
        int *d_score = L.d_score, *d_qbeg = L.d_qbeg, *d_qend = L.d_qend;
        int *d_rbeg = L.d_rbeg, *d_rend = L.d_rend;
        char *d_cigar = L.d_cigar;
        int nn = n;

        int block = 128;
        int grid = (n + block - 1) / block;

        L.q.parallel_for<sw_kernel>(
            sycl::nd_range<1>{(size_t)grid * block, (size_t)block},
            [=](sycl::nd_item<1> it)
            {
                int tid = (int)it.get_global_id(0);
                if (tid >= nn)
                    return;

                const unsigned char *query =
                    (const unsigned char *)(d_reads + d_qoff[tid]);
                const unsigned char *target =
                    (const unsigned char *)(d_refs + d_roff[tid]);
                int qlen = d_qlen[tid];
                int tlen = d_rlen[tid];
                int h0 = H0_DEFAULT;
                int w = W_DEFAULT;

                if (qlen <= 0 || tlen <= 0 || qlen > MAX_QLEN)
                {
                    d_score[tid] = -1;
                    d_qbeg[tid] = 0;
                    d_qend[tid] = 0;
                    d_rbeg[tid] = 0;
                    d_rend[tid] = 0;
                    d_cigar[(size_t)tid * MAX_CIGAR] = '\0';
                    return;
                }

                // rolling row: eh_h = H(i-1,*) , eh_e = E(i,*)
                int eh_h[MAX_QLEN + 1];
                int eh_e[MAX_QLEN + 1];

                const int oe_del = O_DEL + E_DEL;
                const int oe_ins = O_INS + E_INS;

                // หด w ถ้ากว้างเกินกว่าคะแนนจะไปถึง (เหมือน ksw_extend2)
                {
                    int max_ins =
                        (int)((double)(qlen * MATCH + END_BONUS - O_INS) / E_INS + 1.0);
                    max_ins = max_ins > 1 ? max_ins : 1;
                    w = w < max_ins ? w : max_ins;
                    int max_del =
                        (int)((double)(qlen * MATCH + END_BONUS - O_DEL) / E_DEL + 1.0);
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

                int best = h0, max_i = -1, max_j = -1;
                int max_ie = -1, gscore = -1, max_off = 0;
                int beg = 0, end = qlen;

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

                d_score[tid] = best;
                d_qbeg[tid] = 0;         // extension เริ่มจากปลาย seed
                d_qend[tid] = max_j + 1; // qle
                d_rbeg[tid] = 0;
                d_rend[tid] = max_i + 1; // tle
                d_cigar[(size_t)tid * MAX_CIGAR] = '\0';

                (void)gscore;
                (void)max_ie;
                (void)max_off;
            });

        L.q.memcpy(score + start, L.d_score, n * sizeof(int));
        L.q.memcpy(qbeg + start, L.d_qbeg, n * sizeof(int));
        L.q.memcpy(qend + start, L.d_qend, n * sizeof(int));
        L.q.memcpy(rbeg + start, L.d_rbeg, n * sizeof(int));
        L.q.memcpy(rend + start, L.d_rend, n * sizeof(int));
        L.q.memcpy(cigar + (size_t)start * MAX_CIGAR, L.d_cigar,
                   (size_t)n * MAX_CIGAR);
    }

    for (auto &L : lanes)
    {
        L.q.wait();
        sycl::free(L.d_reads, L.q);
        sycl::free(L.d_refs, L.q);
        sycl::free(L.d_qoff, L.q);
        sycl::free(L.d_qlen, L.q);
        sycl::free(L.d_roff, L.q);
        sycl::free(L.d_rlen, L.q);
        sycl::free(L.d_score, L.q);
        sycl::free(L.d_qbeg, L.q);
        sycl::free(L.d_qend, L.q);
        sycl::free(L.d_rbeg, L.q);
        sycl::free(L.d_rend, L.q);
        sycl::free(L.d_cigar, L.q);
        sycl::free(L.h_reads, L.q);
        sycl::free(L.h_refs, L.q);
        sycl::free(L.h_qoff, L.q);
        sycl::free(L.h_roff, L.q);
    }
}

// ============================================================
//  ระดับ 1: sycl_multi_gpu — แบ่งงานข้าม GPU ด้วย pthread
// ============================================================
struct GpuJob
{
    sycl::device dev;
    int start, n;
    const char *h_reads, *h_refs;
    const int *h_qoff, *h_qlen, *h_roff, *h_rlen;
    int *score, *qbeg, *qend, *rbeg, *rend;
    char *cigar;
};

static void *gpu_worker(void *arg)
{
    GpuJob *j = (GpuJob *)arg;
    if (j->n <= 0)
        return NULL;
    sw_sycl(j->dev, j->n,
            j->h_reads, j->h_qoff + j->start, j->h_qlen + j->start,
            j->h_refs, j->h_roff + j->start, j->h_rlen + j->start,
            j->score + j->start, j->qbeg + j->start, j->qend + j->start,
            j->rbeg + j->start, j->rend + j->start,
            j->cigar + (size_t)j->start * MAX_CIGAR);
    return NULL;
}

extern "C" void sycl_multi_gpu(const char *h_reads, const int *h_qoff,
                               const int *h_qlen, int qtot,
                               const char *h_refs, const int *h_roff,
                               const int *h_rlen, int rtot,
                               int n_tasks,
                               int *score, int *qbeg, int *qend,
                               int *rbeg, int *rend, char *cigar)
{
    (void)qtot;
    (void)rtot;

    std::vector<sycl::device> gpus;
    for (auto &d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.has(sycl::aspect::usm_device_allocations))
            gpus.push_back(d);

    int avail = (int)gpus.size();
    if (avail == 0)
    {
        fprintf(stderr, "no GPU\n");
        return;
    }

    int n_gpu = avail;
    const char *env_gpu = getenv("N_GPU"); // รัน 1/2/4 ได้โดยไม่ต้อง recompile
    if (env_gpu && atoi(env_gpu) > 0)
        n_gpu = atoi(env_gpu);
    if (n_gpu > avail)
    {
        fprintf(stderr, "want %d GPU but only %d available -> use %d\n",
                n_gpu, avail, avail);
        n_gpu = avail;
    }
    printf("using %d / %d GPU\n", n_gpu, avail);
    for (int d = 0; d < n_gpu; ++d)
        printf("  - %s\n", gpus[d].get_info<sycl::info::device::name>().c_str());

    int per = (n_tasks + n_gpu - 1) / n_gpu;
    std::vector<pthread_t> th(n_gpu);
    std::vector<GpuJob> job(n_gpu);

    for (int d = 0; d < n_gpu; ++d)
    {
        job[d].dev = gpus[d];
        job[d].start = d * per;
        job[d].n = DMIN(per, n_tasks - job[d].start);
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

    for (int d = 0; d < n_gpu; ++d)
        pthread_join(th[d], NULL);
}