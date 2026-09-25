#include "bwa.h"
#include <stdio.h>

FMIndex *build_fm_index(BWT *bwt)
{
    /*
     * Step 1: count occurrences of each base in BWT
     *   cnt[c] = total occurrences of c in bwt[0..n-1]
     *
     * Step 2: build C[] (cumulative)
     *   C[0] = 0   (nothing is smaller than A=0)
     *   C[c] = C[c-1] + cnt[c-1]
     *   note: sentinel '$' has rank 0 — skip it in C[]
     *         so effectively C counts only A,C,G,T
     *
     * Step 3: build Occ[c][i]  (prefix occurrence table)
     *   Occ[c][0]   = 0
     *   Occ[c][i+1] = Occ[c][i] + (bwt[i] == c ? 1 : 0)
     *
     * Memory: Occ is (ALPHA_SIZE) × (n+2)  uint64_t
     *         for large genomes use sampled Occ (checkpoint every 128bp)
     *         → Phase 1 uses full table, Phase 2 (CUDA) samples it
     */
    FMIndex *fm = malloc(sizeof(FMIndex));
    fm->bwt = bwt;
    uint64_t n = bwt->n;

    /* --- C[] ------------------------------------------------ */
    uint64_t cnt[ALPHA_SIZE] = {0};
    for (uint64_t i = 0; i < n; i++)
        cnt[bwt->bwt[i]]++;

    fm->C[0] = 0;
    for (int c = 1; c < ALPHA_SIZE; c++)
        fm->C[c] = fm->C[c - 1] + cnt[c - 1]; /* TODO: verify off-by-one */

    /* --- Occ[][] ------------------------------------------- */
    fm->Occ = malloc(ALPHA_SIZE * sizeof(uint64_t *));
    for (int c = 0; c < ALPHA_SIZE; c++)
    {
        fm->Occ[c] = calloc(n + 2, sizeof(uint64_t));
        /* TODO: fill prefix sums row by row */
        for (uint64_t i = 0; i < n; i++)
            fm->Occ[c][i + 1] = fm->Occ[c][i] + (bwt->bwt[i] == c ? 1 : 0);
    }

    return fm;
}

/* LF-mapping: LF(i) = C[bwt[i]] + Occ[bwt[i]][i] */
uint64_t lf_map(const FMIndex *fm, uint64_t i)
{
    uint8_t c = fm->bwt->bwt[i];
    return fm->C[c] + fm->Occ[c][i];
}

void lf_roundtrip(const FMIndex *fm)
{
    uint64_t n = fm->bwt->n;
    uint64_t pos = fm->bwt->eof_pos;
    char *recovered = malloc(n);

    for (uint64_t i = n - 1; i > 0; i--)
    {
        recovered[i - 1] = decode_base(fm->bwt->bwt[pos]); /* อ่านก่อน */
        pos = lf_map(fm, pos);                             /* เดินทีหลัง */
    }
    recovered[n - 1] = '\0';
    printf("recovered: %s\n", recovered);
    free(recovered);
}

void fm_free(FMIndex *fm)
{
    if (!fm)
        return;
    for (int c = 0; c < ALPHA_SIZE; c++)
        free(fm->Occ[c]);
    free(fm->Occ);
    free(fm); /* caller frees fm->bwt separately */
}