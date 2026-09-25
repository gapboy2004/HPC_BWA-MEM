#include "bwa.h"
#include <stdio.h>

/* ── encode / decode ──────────────────────────────────── */
uint8_t encode_base(char c)
{
    /* TODO: fill switch statement */
    switch (c)
    {
    case '$':
        return 0;
    case 'A':
        return 1;
    case 'C':
        return 2;
    case 'G':
        return 3;
    case 'T':
        return 4;

    /* add A,C,G,T cases here */
    default:
        fprintf(stderr, "unknown base: %c\n", c);
        return 255;
    }
}

char decode_base(uint8_t b)
{
    /* TODO: fill switch statement */
    const char bases[] = "$ACGT";
    if (b < ALPHA_SIZE)
        return bases[b];
    return '?';
}

/* ── suffix array (naive) ─────────────────────────────── */
/* comparison helper used by qsort */
static const char *_cmp_text;
static int cmp_suffix(const void *a, const void *b)
{
    uint32_t ia = *(uint32_t *)a;
    uint32_t ib = *(uint32_t *)b;
    return strcmp(_cmp_text + ia, _cmp_text + ib);
}

SuffixArray *build_suffix_array(const char *text, uint64_t n)
{
    /*
     * text must already have '$' appended (length n+1)
     * TODO:
     *  1. allocate sa->sa of size n+1
     *  2. initialise sa->sa[i] = i  for i = 0..n
     *  3. sort using qsort + cmp_suffix
     */

    SuffixArray *sa = malloc(sizeof(SuffixArray));
    sa->n = n;
    sa->sa = malloc((n + 1) * sizeof(uint32_t));

    /* init: sa[i] = i  สำหรับ i = 0 .. n */
    for (uint64_t i = 0; i <= n; i++)
        sa->sa[i] = (uint32_t)i;

    /* TODO: initialise and sort */
    /* sort โดยใช้ strcmp เปรียบเทียบ suffix */
    _cmp_text = text;
    qsort(sa->sa, n + 1, sizeof(uint32_t), cmp_suffix);

    return sa;
}

void sa_free(SuffixArray *sa)
{
    if (!sa)
        return;
    free(sa->sa);
    free(sa);
}

/* ── BWT from suffix array ────────────────────────────── */
BWT *build_bwt_from_sa(const char *text, const SuffixArray *sa)
{
    /*
     * bwt[i] = text[ sa->sa[i] - 1 ]
     * if sa->sa[i] == 0  →  bwt[i] = '$'  (sentinel wraps)
     * TODO:
     *  1. allocate bwt->bwt  size n+1
     *  2. loop i = 0..n, compute bwt->bwt[i] via encode_base
     *  3. record eof_pos when sa->sa[i] == 0
     */
    uint64_t n = sa->n;
    BWT *bwt = malloc(sizeof(BWT));
    bwt->n = n + 1;
    bwt->bwt = malloc((n + 1) * sizeof(uint8_t));

    for (uint64_t i = 0; i <= n; i++)
    {
        /* TODO: compute bwt->bwt[i] */
        // ต้อง encode เป็นตัวเลขก่อนเก็บใน bwt->bwt[i] เพราะ CUDA จะใช้ uint8_t แทน char
        if (sa->sa[i] == 0)
        {
            bwt->bwt[i] = encode_base('$');
        }
        else
        {
            bwt->bwt[i] = encode_base(text[sa->sa[i] - 1]);
        }
        (void)i;
    }

    return bwt;
}

void bwt_free(BWT *bwt)
{
    if (!bwt)
        return;
    free(bwt->bwt);
    free(bwt);
}

void reverse_copy(char *dst, const char *src, int len)
{
    for (int i = 0; i < len; i++)
        dst[i] = src[len - 1 - i];

    dst[len] = '\0';
}

LeftExtension make_left_extension(
    const char *query,
    int qlen,
    const char *ref,
    int rlen,
    const ChainSeed *seed)
{
    LeftExtension ext;

    //---------------------------------------
    // Query
    //---------------------------------------

    ext.qlen = seed->qbeg;
    int len = seed->qend - seed->qbeg;
    ext.query = malloc(ext.qlen + 1);

    reverse_copy(
        ext.query,
        query,
        len);

    //---------------------------------------
    // Reference
    //---------------------------------------

    ext.rlen = seed->rpos;

    ext.ref = malloc(ext.rlen + 1);

    reverse_copy(
        ext.ref,
        ref,
        ext.rlen);

    return ext;
}

RightExtension make_right_extension(
    const char *query,
    int qlen,
    const char *ref,
    int rlen,
    const ChainSeed *seed)
{
    RightExtension ext;

    //---------------------------------------
    // Query
    //---------------------------------------

    int qstart = seed->qend;

    ext.qlen = qlen - qstart;

    ext.query = malloc(ext.qlen + 1);

    memcpy(ext.query,
           query + qstart,
           ext.qlen);

    ext.query[ext.qlen] = '\0';

    //---------------------------------------
    // Reference
    //---------------------------------------

    int rstart = seed->rpos + (seed->qend - seed->qbeg);

    ext.rlen = rlen - rstart;

    ext.ref = malloc(ext.rlen + 1);

    memcpy(ext.ref,
           ref + rstart,
           ext.rlen);

    ext.ref[ext.rlen] = '\0';

    return ext;
}

