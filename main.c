#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <ctype.h>
#include <zlib.h>
#include <time.h>

#include "bwa.h"
#include "kseq.h"

#include <stdint.h>
#include "ksw.h"

#ifdef USE_SYCL
    #define run_multi_gpu sycl_multi_gpu
    #define BACKEND_NAME "SYCL"
#else
    #define run_multi_gpu sw_multi_gpu
    #define BACKEND_NAME "CUDA"
#endif

KSEQ_INIT(gzFile, gzread)

static char *h_reads, *h_refs;
static int *h_qoff, *h_qlen, *h_roff, *h_rlen;     // ★ เป็น pointer
static int *score, *qbeg, *qend, *rbeg, *rend;     // ★ เป็น pointer
static char *cigar;
static int max_tasks;                               // ★ ค่า runtime

static uint8_t enc_b(char c)
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
        return 4;
    }
}
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}
//export PATH=/usr/local/cuda/bin:$PATH
int main(int argc, char **argv)
{
    printf("Backend: %s\n", BACKEND_NAME);
    max_tasks = 3000000;                       // default
    const char *e = getenv("MAX_TASKS");
    if (e && atoi(e) > 0) max_tasks = atoi(e);
    if (argc > 1 && atoi(argv[1]) > 0) max_tasks = atoi(argv[1]);
    printf("max_tasks = %d\n", max_tasks);

    h_reads = malloc((size_t)max_tasks * MAX_QLEN);
    h_refs  = malloc((size_t)max_tasks * MAX_RLEN);
    cigar   = malloc((size_t)max_tasks * MAX_CIGAR);

    h_qoff = malloc((size_t)max_tasks * sizeof(int));
    h_qlen = malloc((size_t)max_tasks * sizeof(int));
    h_roff = malloc((size_t)max_tasks * sizeof(int));
    h_rlen = malloc((size_t)max_tasks * sizeof(int));
    score  = malloc((size_t)max_tasks * sizeof(int));
    qbeg   = malloc((size_t)max_tasks * sizeof(int));
    qend   = malloc((size_t)max_tasks * sizeof(int));
    rbeg   = malloc((size_t)max_tasks * sizeof(int));
    rend   = malloc((size_t)max_tasks * sizeof(int));

    if (!h_reads || !h_refs || !h_qoff || !score) {
        fprintf(stderr, "malloc failed (max_tasks=%d)\n", max_tasks);
        return 1;
    }

    // gzFile fq = gzopen("test_prog/query_batch.fasta.gz", "r");
    // gzFile ft = gzopen("test_prog/target_batch.fasta.gz", "r");
    // gzFile fq = gzopen("own_query_batch.fasta", "r");
    // gzFile ft = gzopen("own_target_batch.fasta", "r");
    gzFile fq = gzopen("/home/gapboy/Desktop/real_query.fasta", "r");
    gzFile ft = gzopen("/home/gapboy/Desktop/real_target.fasta", "r");
    if (!fq || !ft)
    {
        fprintf(stderr, "open failed\n");
        return 1;
    }

    kseq_t *kq = kseq_init(fq);
    kseq_t *kt = kseq_init(ft);

    /* ---------- ขั้น 1: เก็บงาน (ไม่คำนวณ) ---------- */
    int n = 0, qcur = 0, rcur = 0, skipped = 0;

    while (kseq_read(kq) >= 0 && kseq_read(kt) >= 0)
    {
        if (n >= max_tasks)
            break;

        int ql = (int)kq->seq.l;
        int rl = (int)kt->seq.l;
        if (ql <= 0 || rl <= 0 || ql > MAX_QLEN || rl > MAX_RLEN)
        {
            skipped++;
            continue;
        }

        h_qoff[n] = qcur;
        h_qlen[n] = ql;
        memcpy(h_reads + qcur, kq->seq.s, ql); /* ★ ต้อง copy: kseq ใช้ buffer ซ้ำ */
        qcur += ql;

        h_roff[n] = rcur;
        h_rlen[n] = rl;
        memcpy(h_refs + rcur, kt->seq.s, rl);
        rcur += rl;

        n++;
    }

    //encode reads and refs to 0-4
    for (int x = 0; x < qcur; ++x) h_reads[x] = enc_b(h_reads[x]);
    for (int x = 0; x < rcur; ++x) h_refs[x]  = enc_b(h_refs[x]);

    kseq_destroy(kq);
    kseq_destroy(kt);
    gzclose(fq);
    gzclose(ft);

    printf("packed %d tasks (skipped %d)\n", n, skipped);
    if (n == 0)
    {
        fprintf(stderr, "no tasks\n");
        return 1;
    }

    /* ---------- ขั้น 2: คำนวณทีเดียวทั้งหมด ---------- */

    double t0 = now_sec();
    // sw_multi_gpu(h_reads, h_qoff, h_qlen, qcur,
    //              h_refs, h_roff, h_rlen, rcur,
    //              n, score, qbeg, qend, rbeg, rend, cigar);

    // sycl_multi_gpu(h_reads, h_qoff, h_qlen, qcur,
    //                h_refs, h_roff, h_rlen, rcur,
    //                n, score, qbeg, qend, rbeg, rend, cigar);

    run_multi_gpu(h_reads, h_qoff, h_qlen, qcur,
              h_refs, h_roff, h_rlen, rcur,
              n, score, qbeg, qend, rbeg, rend, cigar);

    unsigned long long total_cells = 0;
    for (int i = 0; i < n; ++i)
        total_cells += (unsigned)qbeg[i];
    printf("total cells = %llu\n", total_cells);
    /* ---------- ---------------------------------------------- */
    double t1 = now_sec();
    printf("total: %.3f sec  (%d tasks, %.0f tasks/sec)\n",
           t1 - t0, n, n / (t1 - t0));

    float GCUPS = total_cells / ((t1 - t0) * 1e9);
    printf("GCUPS: %.2f\n", GCUPS);
    
    /* ---------- ขั้น 3: หา best จากผลลัพธ์ ---------- */
    // int best_id = 0;
    // for (int i = 1; i < n; i++)
    //     if (score[i] > score[best_id])
    //         best_id = i;

    // printf("\nBest alignment\n");
    // printf("Pair  : %d\n", best_id);
    // printf("Score : %d\n", score[best_id]);
    // printf("CIGAR : %s\n", cigar + (size_t)best_id * MAX_CIGAR);
    // printf("Query : [%d,%d)\n", qbeg[best_id], qend[best_id]);
    // printf("Ref   : [%d,%d)\n", rbeg[best_id], rend[best_id]);

    // // คู่ที่มี mismatch ตอนนี้ควรได้ CIGAR ครบ 150 + มี X
    // int shown = 0;
    // for (int i = 0; i < n && shown < 10; i++)
    // {
    //     if (score[i] < 300)
    //     {
    //         printf("seq%d: score=%d \n", i, score[i]);
    //         shown++;
    //     }
    // }

    return 0;
}