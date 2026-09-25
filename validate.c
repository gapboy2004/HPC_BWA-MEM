      /* ---------- เทียบกับ ksw_extend2 (CPU reference) ---------- */
    {
        int8_t mat[25];
        for (int i = 0, k = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j, ++k)
                mat[k] = (i == 4 || j == 4) ? -1 : (i == j ? 1 : -4);

        uint8_t q[MAX_QLEN], t[MAX_RLEN];
        int n_check = n;
        int bad = 0;

        printf("\n--- GPU vs ksw_extend2 (%d tasks) ---\n", n_check);
        for (int i = 0; i < n_check; ++i) {

            for (int x = 0; x < h_qlen[i]; ++x) q[x] = (uint8_t)h_reads[h_qoff[i] + x];
            for (int x = 0; x < h_rlen[i]; ++x) t[x] = (uint8_t)h_refs [h_roff[i] + x];
            
            int qle, tle, gtle, gscore, max_off;
            int cpu = ksw_extend2(h_qlen[i], q, h_rlen[i], t,
                                  5, mat, 6, 1, 6, 1,   /* o_del e_del o_ins e_ins */
                                  100,                  /* w */
                                  5,                    /* end_bonus */
                                  100,                  /* zdrop */
                                  20,                    /* h0 */
                                  &qle, &tle, &gtle, &gscore, &max_off);

            int ok = (cpu == score[i]) && (qle == qend[i]) && (tle == rend[i]);
              if (!ok) {
                ++bad;
                if (bad <= 10)   // พิมพ์แค่ 10 ตัวแรกที่ผิด
                    printf("task %d gpu(%d,%d,%d) cpu(%d,%d,%d)\n",
                           i, score[i], qend[i], rend[i], cpu, qle, tle);
            }
        }
        printf("--- mismatch: %d / %d ---\n", bad, n_check);
    }

    
        int i = 10;
        printf("task10: qlen=%d rlen=%d\n", h_qlen[i], h_rlen[i]);
        int nq = 0, nr = 0;
        for (int x = 0; x < h_qlen[i]; ++x)
            if (enc_b(h_reads[h_qoff[i] + x]) == 4) ++nq;
        for (int x = 0; x < h_rlen[i]; ++x)
            if (enc_b(h_refs[h_roff[i] + x]) == 4) ++nr;
        printf("task10: N in query=%d, N in ref=%d\n", nq, nr);
        printf("task10 q: %.60s\n", h_reads + h_qoff[i]);
        printf("task10 r: %.60s\n", h_refs + h_roff[i]);