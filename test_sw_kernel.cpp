// test_sw_kernel.cpp — SYCL ล้วน compile ด้วย acpp
//   - sw_cpu()  : CPU reference (band DP, ตรงกับ sw_batch_kernel) = ตัวเฉลย
//   - sw_sycl() : SYCL kernel รันบน GPU จริง = ของที่ port
//   - main()    : ป้อน input เดียวกัน เทียบ score/CIGAR
//
// build:  acpp -O3 --acpp-targets=generic test_sw_kernel.cpp -o test_sw_kernel
// run:    ./test_sw_kernel

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstring>
#include <climits>
#include <string>
#include <vector>
#include <iostream>

// ---- scoring (ต้องตรงกันทั้ง CPU และ GPU) ----
#define MATCH 2
#define MISMATCH -3
#define GAP_OPEN -4
#define GAP_EXT -1

#define STOP 0
#define FROM_DIAG 1
#define FROM_LEFT 2
#define FROM_UP 3

#define BAND_W 1
#define BW (2 * BAND_W + 1) // = 3
#define MAX_QLEN 512
#define MAX_CIGAR 256
#define SLICE ((size_t)(MAX_QLEN + 1) * BW)

#define DMAX(a, b) ((a) > (b) ? (a) : (b))
#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define IABS(x) ((x) < 0 ? -(x) : (x))

// ============================================================
//  ส่วน 1: CPU REFERENCE
//  ถอดจาก sw_batch_kernel (band-packed layout เดียวกับ GPU)
//  ไม่มี tid / pointer offset — รับ read/ref เดี่ยว คืน score+cigar
//  *** ปิด early-stop ให้ตรงกับ CUDA kernel (ที่ comment ทิ้งไว้) ***
// ============================================================
struct Result
{
    int score, qbeg, qend, rbeg, rend;
    char cigar[MAX_CIGAR];
};

static int cpu_append_op(char *buf, int pos, int cnt, char op)
{
    char tmp[12];
    int t = 0;
    if (cnt == 0)
        tmp[t++] = '0';
    while (cnt > 0)
    {
        tmp[t++] = (char)('0' + cnt % 10);
        cnt /= 10;
    }
    for (int k = t - 1; k >= 0; --k)
        buf[pos++] = tmp[k];
    buf[pos++] = op;
    return pos;
}

Result sw_cpu(const char *read, int qlen, const char *ref, int rlen)
{
    Result res;
    // scratch band-packed (เหมือน GPU): (qlen+1) * BW
    std::vector<int> H(SLICE, 0), E(SLICE, 0), F(SLICE, 0), tb(SLICE, 0);

    if (qlen <= 0 || rlen <= 0)
    {
        res.score = 0;
        res.qbeg = res.qend = res.rbeg = res.rend = 0;
        res.cigar[0] = '\0';
        return res;
    }

    for (int k = 0; k < BW; ++k)
    {
        E[k] = -99;
        F[k] = -99;
        tb[k] = -99;
    }

    int best_score = INT_MIN, best_i = qlen, best_j = rlen;
    int w = BAND_W;

    for (int i = 1; i <= qlen; ++i)
    {
        int j_lo = DMAX(1, i - w);
        int j_hi = DMIN(rlen, i + w);

        for (int j = j_lo; j <= j_hi; ++j)
        {
            int k = j - i + w;
            int base = i * BW + k;

            int Eij;
            if (j - 1 >= j_lo)
                Eij = DMAX(H[i * BW + (k - 1)] + GAP_OPEN, E[i * BW + (k - 1)] + GAP_EXT);
            else
                Eij = -98;
            E[base] = Eij;

            int Fij;
            if (IABS((i - 1) - j) <= w)
                Fij = DMAX(H[(i - 1) * BW + (k + 1)] + GAP_OPEN, F[(i - 1) * BW + (k + 1)] + GAP_EXT);
            else
                Fij = -98;
            F[base] = Fij;

            int sub = (read[i - 1] == ref[j - 1]) ? MATCH : MISMATCH;
            int diag = H[(i - 1) * BW + k] + sub;

            int h = DMAX(DMAX(DMAX(diag, Eij), Fij), 0);
            H[base] = h;

            int t;
            if (h == 0)
                t = STOP;
            else if (h == diag)
                t = FROM_DIAG;
            else if (h == Eij)
                t = FROM_LEFT;
            else
                t = FROM_UP;
            tb[base] = t;

            if (h > best_score)
            {
                best_score = h;
                best_i = i;
                best_j = j;
            }
        }
    }

    // traceback (band-packed)
    char raw[MAX_CIGAR];
    int clen = 0;
    int i = best_i, j = best_j;
    while (i > 0 && j > 0 && clen < MAX_CIGAR)
    {
        int k = j - i + w;
        if (k < 0 || k >= BW)
            break;
        if (H[i * BW + k] <= 0)
            break;
        int t = tb[i * BW + k];
        if (t == FROM_DIAG)
        {
            raw[clen++] = (read[i - 1] == ref[j - 1]) ? 'M' : 'X';
            --i;
            --j;
        }
        else if (t == FROM_LEFT)
        {
            raw[clen++] = 'D';
            --j;
        }
        else
        {
            raw[clen++] = 'I';
            --i;
        }
    }

    res.score = best_score;
    res.qbeg = i;
    res.qend = best_i;
    res.rbeg = j;
    res.rend = best_j;

    int ci = 0;
    for (int k = clen - 1; k >= 0;)
    {
        char op = raw[k];
        int cnt = 0;
        while (k >= 0 && raw[k] == op)
        {
            ++cnt;
            --k;
        }
        ci = cpu_append_op(res.cigar, ci, cnt, op);
    }
    res.cigar[ci] = '\0';
    return res;
}

// ============================================================
//  ส่วน 2: SYCL KERNEL — รันบน GPU จริง
//  port ตรงจาก sw_batch_kernel: 1 work-item = 1 task
// ============================================================
class sw_kernel;

void sw_sycl(sycl::queue &q, int n_tasks,
             const char *h_reads, const int *h_qoff, const int *h_qlen,
             const char *h_refs, const int *h_roff, const int *h_rlen,
             int qtot, int rtot,
             int *h_score, int *h_qbeg, int *h_qend, int *h_rbeg, int *h_rend,
             char *h_cigar)
{
    // ---- จอง device memory (USM) ----
    char *d_reads = sycl::malloc_device<char>(qtot, q);
    char *d_refs = sycl::malloc_device<char>(rtot, q);
    int *d_qoff = sycl::malloc_device<int>(n_tasks, q);
    int *d_qlen = sycl::malloc_device<int>(n_tasks, q);
    int *d_roff = sycl::malloc_device<int>(n_tasks, q);
    int *d_rlen = sycl::malloc_device<int>(n_tasks, q);

    int *d_H = sycl::malloc_device<int>(SLICE * n_tasks, q);
    int *d_E = sycl::malloc_device<int>(SLICE * n_tasks, q);
    int *d_F = sycl::malloc_device<int>(SLICE * n_tasks, q);
    int *d_tb = sycl::malloc_device<int>(SLICE * n_tasks, q);

    int *d_score = sycl::malloc_device<int>(n_tasks, q);
    int *d_qbeg = sycl::malloc_device<int>(n_tasks, q);
    int *d_qend = sycl::malloc_device<int>(n_tasks, q);
    int *d_rbeg = sycl::malloc_device<int>(n_tasks, q);
    int *d_rend = sycl::malloc_device<int>(n_tasks, q);
    char *d_cigar = sycl::malloc_device<char>((size_t)n_tasks * MAX_CIGAR, q);

    // ---- copy input เข้า device ----
    q.memcpy(d_reads, h_reads, qtot);
    q.memcpy(d_refs, h_refs, rtot);
    q.memcpy(d_qoff, h_qoff, n_tasks * sizeof(int));
    q.memcpy(d_qlen, h_qlen, n_tasks * sizeof(int));
    q.memcpy(d_roff, h_roff, n_tasks * sizeof(int));
    q.memcpy(d_rlen, h_rlen, n_tasks * sizeof(int));

    // scratch = 0 (แทน calloc)
    q.memset(d_H, 0, SLICE * n_tasks * sizeof(int));
    q.memset(d_E, 0, SLICE * n_tasks * sizeof(int));
    q.memset(d_F, 0, SLICE * n_tasks * sizeof(int));
    q.memset(d_tb, 0, SLICE * n_tasks * sizeof(int));

    // ---- launch: 1 work-item = 1 task ----
    int block = 128;
    int grid = (n_tasks + block - 1) / block;
    q.parallel_for<sw_kernel>(
        sycl::nd_range<1>{(size_t)grid * block, (size_t)block},
        [=](sycl::nd_item<1> it)
        {
            int tid = it.get_global_id(0);
            if (tid >= n_tasks)
                return;

            const char *read = d_reads + d_qoff[tid];
            const char *ref = d_refs + d_roff[tid];
            int qlen = d_qlen[tid];
            int rlen = d_rlen[tid];

            int *H = d_H + (size_t)tid * SLICE;
            int *E = d_E + (size_t)tid * SLICE;
            int *F = d_F + (size_t)tid * SLICE;
            int *tb = d_tb + (size_t)tid * SLICE;

            if (qlen <= 0 || rlen <= 0)
            {
                d_score[tid] = 0;
                d_qbeg[tid] = 0;
                d_qend[tid] = 0;
                d_rbeg[tid] = 0;
                d_rend[tid] = 0;
                d_cigar[(size_t)tid * MAX_CIGAR] = '\0';
                return;
            }

            for (int k = 0; k < BW; ++k)
            {
                E[k] = -99;
                F[k] = -99;
                tb[k] = -99;
            }

            int best_score = INT_MIN, best_i = qlen, best_j = rlen;
            int w = BAND_W;

            for (int i = 1; i <= qlen; ++i)
            {
                int j_lo = DMAX(1, i - w);
                int j_hi = DMIN(rlen, i + w);

                for (int j = j_lo; j <= j_hi; ++j)
                {
                    int k = j - i + w;
                    int base = i * BW + k;

                    int Eij;
                    if (j - 1 >= j_lo)
                        Eij = DMAX(H[i * BW + (k - 1)] + GAP_OPEN, E[i * BW + (k - 1)] + GAP_EXT);
                    else
                        Eij = -98;
                    E[base] = Eij;

                    int Fij;
                    if (IABS((i - 1) - j) <= w)
                        Fij = DMAX(H[(i - 1) * BW + (k + 1)] + GAP_OPEN, F[(i - 1) * BW + (k + 1)] + GAP_EXT);
                    else
                        Fij = -98;
                    F[base] = Fij;

                    int sub = (read[i - 1] == ref[j - 1]) ? MATCH : MISMATCH;
                    int diag = H[(i - 1) * BW + k] + sub;

                    int h = DMAX(DMAX(DMAX(diag, Eij), Fij), 0);
                    H[base] = h;

                    int t;
                    if (h == 0)
                        t = STOP;
                    else if (h == diag)
                        t = FROM_DIAG;
                    else if (h == Eij)
                        t = FROM_LEFT;
                    else
                        t = FROM_UP;
                    tb[base] = t;

                    if (h > best_score)
                    {
                        best_score = h;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            char raw[MAX_CIGAR];
            int clen = 0;
            int i = best_i, j = best_j;
            while (i > 0 && j > 0 && clen < MAX_CIGAR)
            {
                int k = j - i + w;
                if (k < 0 || k >= BW)
                    break;
                if (H[i * BW + k] <= 0)
                    break;
                int t = tb[i * BW + k];
                if (t == FROM_DIAG)
                {
                    raw[clen++] = (read[i - 1] == ref[j - 1]) ? 'M' : 'X';
                    --i;
                    --j;
                }
                else if (t == FROM_LEFT)
                {
                    raw[clen++] = 'D';
                    --j;
                }
                else
                {
                    raw[clen++] = 'I';
                    --i;
                }
            }

            d_score[tid] = best_score;
            d_qbeg[tid] = i;
            d_qend[tid] = best_i;
            d_rbeg[tid] = j;
            d_rend[tid] = best_j;

            char *cig = d_cigar + (size_t)tid * MAX_CIGAR;
            int ci = 0;
            for (int k = clen - 1; k >= 0;)
            {
                char op = raw[k];
                int cnt = 0;
                while (k >= 0 && raw[k] == op)
                {
                    ++cnt;
                    --k;
                }
                // inline append_op
                char tmp[12];
                int tt = 0;
                if (cnt == 0)
                    tmp[tt++] = '0';
                while (cnt > 0)
                {
                    tmp[tt++] = (char)('0' + cnt % 10);
                    cnt /= 10;
                }
                for (int z = tt - 1; z >= 0; --z)
                    cig[ci++] = tmp[z];
                cig[ci++] = op;
            }
            cig[ci] = '\0';
        });
    q.wait();

    // ---- copy ผลกลับ host ----
    q.memcpy(h_score, d_score, n_tasks * sizeof(int));
    q.memcpy(h_qbeg, d_qbeg, n_tasks * sizeof(int));
    q.memcpy(h_qend, d_qend, n_tasks * sizeof(int));
    q.memcpy(h_rbeg, d_rbeg, n_tasks * sizeof(int));
    q.memcpy(h_rend, d_rend, n_tasks * sizeof(int));
    q.memcpy(h_cigar, d_cigar, (size_t)n_tasks * MAX_CIGAR);
    q.wait();

    sycl::free(d_reads, q);
    sycl::free(d_refs, q);
    sycl::free(d_qoff, q);
    sycl::free(d_qlen, q);
    sycl::free(d_roff, q);
    sycl::free(d_rlen, q);
    sycl::free(d_H, q);
    sycl::free(d_E, q);
    sycl::free(d_F, q);
    sycl::free(d_tb, q);
    sycl::free(d_score, q);
    sycl::free(d_qbeg, q);
    sycl::free(d_qend, q);
    sycl::free(d_rbeg, q);
    sycl::free(d_rend, q);
    sycl::free(d_cigar, q);
}

// ============================================================
//  ส่วน 3: main — เทียบ CPU vs SYCL(GPU)
// ============================================================
struct TestCase
{
    std::string read, ref;
};

int main()
{
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::cout << "GPU: " << q.get_device().get_info<sycl::info::device::name>() << "\n\n";

    // test cases 3 ระดับ
    std::vector<TestCase> tests = {
        {"ACGTACGT", "ACGTACGT"},                 // เหมือนกันเป๊ะ
        {"ACGTACGT", "ACGAACGT"},                 // 1 mismatch
        {"ACGT", "ACGT"},                         // สั้น
        {"GGGGCCCC", "GGGGCCCC"},                 // เหมือนกัน อีกชุด
        {"ACGTACGT", "ACGTCGT"},                  // deletion (ref สั้นกว่า)
        {"ACGTACGT", "ACGTAACGT"},                // insertion (ref ยาวกว่า)
        {"AAAATTTT", "TTTTAAAA"},                 // ไม่ match เลย → score ต่ำ/CIGAR สั้น
        {"ACGTACGTACGTACGT", "ACGTACGTACGTACGT"}, // ยาวขึ้น (16)
    };
    int n = tests.size();

    // แพ็คงานเป็น flat array (เหมือน main.c จริง)
    std::vector<char> reads, refs;
    std::vector<int> qoff(n), qlen(n), roff(n), rlen(n);
    for (int i = 0; i < n; ++i)
    {
        qoff[i] = reads.size();
        qlen[i] = tests[i].read.size();
        reads.insert(reads.end(), tests[i].read.begin(), tests[i].read.end());
        roff[i] = refs.size();
        rlen[i] = tests[i].ref.size();
        refs.insert(refs.end(), tests[i].ref.begin(), tests[i].ref.end());
    }

    // รัน SYCL บน GPU
    std::vector<int> s_score(n), s_qbeg(n), s_qend(n), s_rbeg(n), s_rend(n);
    std::vector<char> s_cigar((size_t)n * MAX_CIGAR);
    sw_sycl(q, n, reads.data(), qoff.data(), qlen.data(),
            refs.data(), roff.data(), rlen.data(),
            reads.size(), refs.size(),
            s_score.data(), s_qbeg.data(), s_qend.data(),
            s_rbeg.data(), s_rend.data(), s_cigar.data());

    // เทียบทีละเคส
    int pass = 0;
    for (int i = 0; i < n; ++i)
    {
        Result cpu = sw_cpu(tests[i].read.c_str(), tests[i].read.size(),
                            tests[i].ref.c_str(), tests[i].ref.size());
        const char *sycl_cig = s_cigar.data() + (size_t)i * MAX_CIGAR;

        bool ok = (cpu.score == s_score[i]) &&
                  (std::strcmp(cpu.cigar, sycl_cig) == 0);
        if (ok)
            ++pass;

        printf("Test %d: %s x %s\n", i, tests[i].read.c_str(), tests[i].ref.c_str());
        printf("  CPU : score=%d cigar=%s qbeg=%d qend=%d rbeg=%d rend=%d\n",
               cpu.score, cpu.cigar, cpu.qbeg, cpu.qend, cpu.rbeg, cpu.rend);
        printf("  SYCL: score=%d cigar=%s qbeg=%d qend=%d rbeg=%d rend=%d\n",
               s_score[i], sycl_cig, s_qbeg[i], s_qend[i], s_rbeg[i], s_rend[i]);
        printf("  %s\n\n", ok ? "PASS" : "FAIL <<<<<");
    }

    printf("==== %d/%d ผ่าน ====\n", pass, n);
    return (pass == n) ? 0 : 1;
}