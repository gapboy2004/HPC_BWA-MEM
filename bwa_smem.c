#include "bwa.h"
#include <stdio.h>

/* ค้นหา pattern ใน genome โดยไม่ต้อง scan ทั้ง genome */
void backward_search(const FMIndex *fm,
                     const char *pattern,
                     uint64_t m,   /* ความยาว pattern */
                     uint64_t *lo, /* output: range เริ่ม */
                     uint64_t *hi)
{ 
    *lo = 0;
    *hi = fm->bwt->n - 1;
    for (int64_t i = m - 1; i >= 0; i--)
    {
        uint8_t c = encode_base(pattern[i]);
        *lo = fm->C[c] + fm->Occ[c][*lo];
        *hi = fm->C[c] + fm->Occ[c][*hi + 1] - 1;
        if (*lo > *hi)
        {
            /* pattern ไม่พบใน genome */
            *lo = *hi = UINT64_MAX;
            return;
        }
    }
}
int find_smems(const FMIndex *fm, const char *read, uint64_t m, SMEM *smems)
{
    int n_smems = 0;
    // เราต้องถอยหลังจากตัวสุดท้ายของ read ย้อนกลับมา
    // นี่คือการทำ Backward search แบบมาตรฐาน
    uint64_t i = m; 
    while (i > 0)
    {
        uint64_t lo = 0;
        uint64_t hi = fm->bwt->n - 1;
        uint64_t j = i - 1; /* เริ่มจากตัวสุดท้าย */
        
        // ขยายย้อนไปทางซ้าย
        while (j < m) 
        {
            uint8_t c = encode_base(read[j]);
            if (c >= ALPHA_SIZE) break;

            uint64_t nlo = fm->C[c] + fm->Occ[c][lo];
            uint64_t nhi = fm->C[c] + fm->Occ[c][hi + 1] - 1;

            if (nlo > nhi) break;

            lo = nlo;
            hi = nhi;
            if (j == 0) break; // ถึงตัวแรกของ read แล้ว
            j--;
        }

        // หาก match ได้ ให้บันทึก
        if (i - j > 0) {
            smems[n_smems].qbeg = j;
            smems[n_smems].qend = i;
            smems[n_smems].lo = lo;
            smems[n_smems].hi = hi;
            n_smems++;
            i = j; // กระโดดข้าม match ที่เจอไป
        } else {
            i--;
        }
    }
    return n_smems;
}

#define MAX_GAP 100
#define GAP_PENALTY 1

int chain_seeds(const uint32_t *SA,
                SMEM *smems,
                int n_smems,
                ChainSeed *seeds)
{
    int n_seeds = 0;
    
    /* Step 1: สร้าง Seeds จาก SMEMs */
    for (int i = 0; i < n_smems; i++) {
        for (uint64_t r = smems[i].lo; r <= smems[i].hi; r++) {
            seeds[n_seeds].rpos = SA[r]; 
            seeds[n_seeds].qbeg = smems[i].qbeg;
            seeds[n_seeds].qend = smems[i].qend;
            seeds[n_seeds].score = (int)(smems[i].qend - smems[i].qbeg);
            seeds[n_seeds].prev = -1;
            n_seeds++;
        }
    }

    /* Step 2: เรียง Seed ตามตำแหน่งใน Ref (rpos) เพื่อเตรียมทำ DP */
    for (int i = 0; i < n_seeds - 1; i++) {
        for (int j = i + 1; j < n_seeds; j++) {
            if (seeds[j].rpos < seeds[i].rpos) {
                ChainSeed tmp = seeds[i];
                seeds[i] = seeds[j];
                seeds[j] = tmp;
            }
        }
    }

    /* Step 3: DP หา chain score */
    for (int i = 1; i < n_seeds; i++) {
        for (int j = 0; j < i; j++) {
            // เงื่อนไข: Seed ต้องเรียงลำดับไปข้างหน้าในทั้ง Query และ Ref
            if (seeds[j].qbeg >= seeds[i].qbeg || seeds[j].rpos >= seeds[i].rpos)
                continue;

            uint64_t gap = seeds[i].rpos - seeds[j].rpos;
            if (gap > MAX_GAP) continue;

            int len_i = (int)(seeds[i].qend - seeds[i].qbeg);
            int penalty = (int)gap * GAP_PENALTY;
            int new_score = seeds[j].score + len_i - penalty;

            if (new_score > seeds[i].score) {
                seeds[i].score = new_score;
                seeds[i].prev = j;
            }
        }
    }
    return n_seeds;
}