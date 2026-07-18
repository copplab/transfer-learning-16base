"""
DNA Feature Generator

Two functions:
- create_feature_vectors: For training (10-base sequences with color labels)
- create_feature_vectors2: For prediction (16-base sequences, normalized)
"""

import pandas as pd

# =============================================================================
# Feature Column Names
# =============================================================================
FEATURE_COLUMNS = [
    'a_0a', 'a_1a', 'a_2a', 'a_3a', 'a_4a', 'a_5a', 'a_6a', 'a_7a', 'a_8a',
    'c_0c', 'c_1c', 'c_2c', 'c_3c', 'c_4c', 'c_5c', 'c_6c', 'c_7c', 'c_8c',
    'g_0g', 'g_1g', 'g_2g', 'g_3g', 'g_4g', 'g_5g', 'g_6g', 'g_7g', 'g_8g',
    't_0t', 't_1t', 't_2t', 't_3t', 't_4t', 't_5t', 't_6t', 't_7t', 't_8t',
    'a_0c', 'a_1c', 'a_2c', 'a_3c', 'a_4c', 'a_5c', 'a_6c', 'a_7c', 'a_8c',
    'a_0g', 'a_1g', 'a_2g', 'a_3g', 'a_4g', 'a_5g', 'a_6g', 'a_7g', 'a_8g',
    'a_0t', 'a_1t', 'a_2t', 'a_3t', 'a_4t', 'a_5t', 'a_6t', 'a_7t', 'a_8t',
    'c_0a', 'c_1a', 'c_2a', 'c_3a', 'c_4a', 'c_5a', 'c_6a', 'c_7a', 'c_8a',
    'c_0g', 'c_1g', 'c_2g', 'c_3g', 'c_4g', 'c_5g', 'c_6g', 'c_7g', 'c_8g',
    'c_0t', 'c_1t', 'c_2t', 'c_3t', 'c_4t', 'c_5t', 'c_6t', 'c_7t', 'c_8t',
    'g_0a', 'g_1a', 'g_2a', 'g_3a', 'g_4a', 'g_5a', 'g_6a', 'g_7a', 'g_8a',
    'g_0c', 'g_1c', 'g_2c', 'g_3c', 'g_4c', 'g_5c', 'g_6c', 'g_7c', 'g_8c',
    'g_0t', 'g_1t', 'g_2t', 'g_3t', 'g_4t', 'g_5t', 'g_6t', 'g_7t', 'g_8t',
    't_0a', 't_1a', 't_2a', 't_3a', 't_4a', 't_5a', 't_6a', 't_7a', 't_8a',
    't_0c', 't_1c', 't_2c', 't_3c', 't_4c', 't_5c', 't_6c', 't_7c', 't_8c',
    't_0g', 't_1g', 't_2g', 't_3g', 't_4g', 't_5g', 't_6g', 't_7g', 't_8g'
]


# =============================================================================
# Training: 10-base sequences with color labels
# =============================================================================
def create_feature_vectors(inFile):
    vectors = []
    with open(inFile, 'r') as f:
        for line in f:
            list = []

            aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gg0, gg1, gg2, gg3, gg4, gg5, gg6, gg7, gg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            at0, at1, at2, at3, at4, at5, at6, at7, at8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ca0, ca1, ca2, ca3, ca4, ca5, ca6, ca7, ca8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cg0, cg1, cg2, cg3, cg4, cg5, cg6, cg7, cg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ga0, ga1, ga2, ga3, ga4, ga5, ga6, ga7, ga8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gc0, gc1, gc2, gc3, gc4, gc5, gc6, gc7, gc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gt0, gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7, tg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            g0, g1, g2, g3, g4, g5, g6, g7, g8, g9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            words = line.split(" ")

            for i in range(len(words)):
                if (words[i] == "A"):
                    if (i == 0):
                        a0 += 1
                    elif (i == 1):
                        a1 = 1
                    elif (i == 2):
                        a2 += 1
                    elif (i == 3):
                        a3 += 1
                    elif (i == 4):
                        a4 += 1
                    elif (i == 5):
                        a5 += 1
                    elif (i == 6):
                        a6 += 1
                    elif (i == 7):
                        a7 += 1
                    elif (i == 8):
                        a8 += 1
                    elif (i == 9):
                        a9 += 1
            list.append(float(words[10]))
            for i in range(len(words)):
                if (words[i] == "C"):
                    if (i == 0):
                        c0 += 1
                    elif (i == 1):
                        c1 += 1
                    elif (i == 2):
                        c2 += 1
                    elif (i == 3):
                        c3 += 1
                    elif (i == 4):
                        c4 += 1
                    elif (i == 5):
                        c5 += 1
                    elif (i == 6):
                        c6 += 1
                    elif (i == 7):
                        c7 += 1
                    elif (i == 8):
                        c8 += 1
                    elif (i == 9):
                        c9 += 1
            for i in range(len(words)):
                if (words[i] == "G"):
                    if (i == 0):
                        g0 += 1
                    elif (i == 1):
                        g1 += 1
                    elif (i == 2):
                        g2 += 1
                    elif (i == 3):
                        g3 += 1
                    elif (i == 4):
                        g4 += 1
                    elif (i == 5):
                        g5 += 1
                    elif (i == 6):
                        g6 += 1
                    elif (i == 7):
                        g7 += 1
                    elif (i == 8):
                        g8 += 1
                    elif (i == 9):
                        g9 += 1
            for i in range(len(words)):
                if (words[i] == "T"):
                    if (i == 0):
                        t0 += 1
                    elif (i == 1):
                        t1 += 1
                    elif (i == 2):
                        t2 += 1
                    elif (i == 3):
                        t3 += 1
                    elif (i == 4):
                        t4 += 1
                    elif (i == 5):
                        t5 += 1
                    elif (i == 6):
                        t6 += 1
                    elif (i == 7):
                        t7 += 1
                    elif (i == 8):
                        t8 += 1
                    elif (i == 9):
                        t9 += 1

            # Distance 1
            for i, j in zip(words[::], words[1::]):
                if (i == "A") and (j == "A"):
                    aa0 = aa0 + 1
                elif (i == "C") and (j == "C"):
                    cc0 = cc0 + 1
                elif (i == "G") and (j == "G"):
                    gg0 = gg0 + 1
                elif (i == "T") and (j == "T"):
                    tt0 = tt0 + 1
                elif (i == "A") and (j == "C"):
                    ac0 = ac0 + 1
                elif (i == "A") and (j == "G"):
                    ag0 = ag0 + 1
                elif (i == "A") and (j == "T"):
                    at0 = at0 + 1
                elif (i == "C") and (j == "A"):
                    ca0 = ca0 + 1
                elif (i == "C") and (j == "G"):
                    cg0 = cg0 + 1
                elif (i == "C") and (j == "T"):
                    ct0 = ct0 + 1
                elif (i == "G") and (j == "A"):
                    ga0 = ga0 + 1
                elif (i == "G") and (j == "C"):
                    gc0 = gc0 + 1
                elif (i == "G") and (j == "T"):
                    gt0 = gt0 + 1
                elif (i == "T") and (j == "A"):
                    ta0 = ta0 + 1
                elif (i == "T") and (j == "C"):
                    tc0 = tc0 + 1
                elif (i == "T") and (j == "G"):
                    tg0 = tg0 + 1

            # Distance 2
            for i, j in zip(words[::], words[2::]):
                if (i == "A") and (j == "A"):
                    aa1 = aa1 + 1
                elif (i == "C") and (j == "C"):
                    cc1 = cc1 + 1
                elif (i == "G") and (j == "G"):
                    gg1 = gg1 + 1
                elif (i == "T") and (j == "T"):
                    tt1 = tt1 + 1
                elif (i == "A") and (j == "C"):
                    ac1 = ac1 + 1
                elif (i == "A") and (j == "G"):
                    ag1 = ag1 + 1
                elif (i == "A") and (j == "T"):
                    at1 = at1 + 1
                elif (i == "C") and (j == "A"):
                    ca1 = ca1 + 1
                elif (i == "C") and (j == "G"):
                    cg1 = cg1 + 1
                elif (i == "C") and (j == "T"):
                    ct1 = ct1 + 1
                elif (i == "G") and (j == "A"):
                    ga1 = ga1 + 1
                elif (i == "G") and (j == "C"):
                    gc1 = gc1 + 1
                elif (i == "G") and (j == "T"):
                    gt1 = gt1 + 1
                elif (i == "T") and (j == "A"):
                    ta1 = ta1 + 1
                elif (i == "T") and (j == "C"):
                    tc1 = tc1 + 1
                elif (i == "T") and (j == "G"):
                    tg1 = tg1 + 1

            # Distance 3
            for i, j in zip(words[::], words[3::]):
                if (i == "A") and (j == "A"):
                    aa2 = aa2 + 1
                elif (i == "C") and (j == "C"):
                    cc2 = cc2 + 1
                elif (i == "G") and (j == "G"):
                    gg2 = gg2 + 1
                elif (i == "T") and (j == "T"):
                    tt2 = tt2 + 1
                elif (i == "A") and (j == "C"):
                    ac2 = ac2 + 1
                elif (i == "A") and (j == "G"):
                    ag2 = ag2 + 1
                elif (i == "A") and (j == "T"):
                    at2 = at2 + 1
                elif (i == "C") and (j == "A"):
                    ca2 = ca2 + 1
                elif (i == "C") and (j == "G"):
                    cg2 = cg2 + 1
                elif (i == "C") and (j == "T"):
                    ct2 = ct2 + 1
                elif (i == "G") and (j == "A"):
                    ga2 = ga2 + 1
                elif (i == "G") and (j == "C"):
                    gc2 = gc2 + 1
                elif (i == "G") and (j == "T"):
                    gt2 = gt2 + 1
                elif (i == "T") and (j == "A"):
                    ta2 = ta2 + 1
                elif (i == "T") and (j == "C"):
                    tc2 = tc2 + 1
                elif (i == "T") and (j == "G"):
                    tg2 = tg2 + 1

            # Distance 4
            for i, j in zip(words[::], words[4::]):
                if (i == "A") and (j == "A"):
                    aa3 = aa3 + 1
                elif (i == "C") and (j == "C"):
                    cc3 = cc3 + 1
                elif (i == "G") and (j == "G"):
                    gg3 = gg3 + 1
                elif (i == "T") and (j == "T"):
                    tt3 = tt3 + 1
                elif (i == "A") and (j == "C"):
                    ac3 = ac3 + 1
                elif (i == "A") and (j == "G"):
                    ag3 = ag3 + 1
                elif (i == "A") and (j == "T"):
                    at3 = at3 + 1
                elif (i == "C") and (j == "A"):
                    ca3 = ca3 + 1
                elif (i == "C") and (j == "G"):
                    cg3 = cg3 + 1
                elif (i == "C") and (j == "T"):
                    ct3 = ct3 + 1
                elif (i == "G") and (j == "A"):
                    ga3 = ga3 + 1
                elif (i == "G") and (j == "C"):
                    gc3 = gc3 + 1
                elif (i == "G") and (j == "T"):
                    gt3 = gt3 + 1
                elif (i == "T") and (j == "A"):
                    ta3 = ta3 + 1
                elif (i == "T") and (j == "C"):
                    tc3 = tc3 + 1
                elif (i == "T") and (j == "G"):
                    tg3 = tg3 + 1

            # Distance 5
            for i, j in zip(words[::], words[5::]):
                if (i == "A") and (j == "A"):
                    aa4 = aa4 + 1
                elif (i == "C") and (j == "C"):
                    cc4 = cc4 + 1
                elif (i == "G") and (j == "G"):
                    gg4 = gg4 + 1
                elif (i == "T") and (j == "T"):
                    tt4 = tt4 + 1
                elif (i == "A") and (j == "C"):
                    ac4 = ac4 + 1
                elif (i == "A") and (j == "G"):
                    ag4 = ag4 + 1
                elif (i == "A") and (j == "T"):
                    at4 = at4 + 1
                elif (i == "C") and (j == "A"):
                    ca4 = ca4 + 1
                elif (i == "C") and (j == "G"):
                    cg4 = cg4 + 1
                elif (i == "C") and (j == "T"):
                    ct4 = ct4 + 1
                elif (i == "G") and (j == "A"):
                    ga4 = ga4 + 1
                elif (i == "G") and (j == "C"):
                    gc4 = gc4 + 1
                elif (i == "G") and (j == "T"):
                    gt4 = gt4 + 1
                elif (i == "T") and (j == "A"):
                    ta4 = ta4 + 1
                elif (i == "T") and (j == "C"):
                    tc4 = tc4 + 1
                elif (i == "T") and (j == "G"):
                    tg4 = tg4 + 1

            # Distance 6
            for i, j in zip(words[::], words[6::]):
                if (i == "A") and (j == "A"):
                    aa5 = aa5 + 1
                elif (i == "C") and (j == "C"):
                    cc5 = cc5 + 1
                elif (i == "G") and (j == "G"):
                    gg5 = gg5 + 1
                elif (i == "T") and (j == "T"):
                    tt5 = tt5 + 1
                elif (i == "A") and (j == "C"):
                    ac5 = ac5 + 1
                elif (i == "A") and (j == "G"):
                    ag5 = ag5 + 1
                elif (i == "A") and (j == "T"):
                    at5 = at5 + 1
                elif (i == "C") and (j == "A"):
                    ca5 = ca5 + 1
                elif (i == "C") and (j == "G"):
                    cg5 = cg5 + 1
                elif (i == "C") and (j == "T"):
                    ct5 = ct5 + 1
                elif (i == "G") and (j == "A"):
                    ga5 = ga5 + 1
                elif (i == "G") and (j == "C"):
                    gc5 = gc5 + 1
                elif (i == "G") and (j == "T"):
                    gt5 = gt5 + 1
                elif (i == "T") and (j == "A"):
                    ta5 = ta5 + 1
                elif (i == "T") and (j == "C"):
                    tc5 = tc5 + 1
                elif (i == "T") and (j == "G"):
                    tg5 = tg5 + 1

            # Distance 7
            for i, j in zip(words[::], words[7::]):
                if (i == "A") and (j == "A"):
                    aa6 = aa6 + 1
                elif (i == "C") and (j == "C"):
                    cc6 = cc6 + 1
                elif (i == "G") and (j == "G"):
                    gg6 = gg6 + 1
                elif (i == "T") and (j == "T"):
                    tt6 = tt6 + 1
                elif (i == "A") and (j == "C"):
                    ac6 = ac6 + 1
                elif (i == "A") and (j == "G"):
                    ag6 = ag6 + 1
                elif (i == "A") and (j == "T"):
                    at6 = at6 + 1
                elif (i == "C") and (j == "A"):
                    ca6 = ca6 + 1
                elif (i == "C") and (j == "G"):
                    cg6 = cg6 + 1
                elif (i == "C") and (j == "T"):
                    ct6 = ct6 + 1
                elif (i == "G") and (j == "A"):
                    ga6 = ga6 + 1
                elif (i == "G") and (j == "C"):
                    gc6 = gc6 + 1
                elif (i == "G") and (j == "T"):
                    gt6 = gt6 + 1
                elif (i == "T") and (j == "A"):
                    ta6 = ta6 + 1
                elif (i == "T") and (j == "C"):
                    tc6 = tc6 + 1
                elif (i == "T") and (j == "G"):
                    tg6 = tg6 + 1

            # Distance 8
            for i, j in zip(words[::], words[8::]):
                if (i == "A") and (j == "A"):
                    aa7 = aa7 + 1
                elif (i == "C") and (j == "C"):
                    cc7 = cc7 + 1
                elif (i == "G") and (j == "G"):
                    gg7 = gg7 + 1
                elif (i == "T") and (j == "T"):
                    tt7 = tt7 + 1
                elif (i == "A") and (j == "C"):
                    ac7 = ac7 + 1
                elif (i == "A") and (j == "G"):
                    ag7 = ag7 + 1
                elif (i == "A") and (j == "T"):
                    at7 = at7 + 1
                elif (i == "C") and (j == "A"):
                    ca7 = ca7 + 1
                elif (i == "C") and (j == "G"):
                    cg7 = cg7 + 1
                elif (i == "C") and (j == "T"):
                    ct7 = ct7 + 1
                elif (i == "G") and (j == "A"):
                    ga7 = ga7 + 1
                elif (i == "G") and (j == "C"):
                    gc7 = gc7 + 1
                elif (i == "G") and (j == "T"):
                    gt7 = gt7 + 1
                elif (i == "T") and (j == "A"):
                    ta7 = ta7 + 1
                elif (i == "T") and (j == "C"):
                    tc7 = tc7 + 1
                elif (i == "T") and (j == "G"):
                    tg7 = tg7 + 1

            # Distance 9
            for i, j in zip(words[::], words[9::]):
                if (i == "A") and (j == "A"):
                    aa8 = aa8 + 1
                elif (i == "C") and (j == "C"):
                    cc8 = cc8 + 1
                elif (i == "G") and (j == "G"):
                    gg8 = gg8 + 1
                elif (i == "T") and (j == "T"):
                    tt8 = tt8 + 1
                elif (i == "A") and (j == "C"):
                    ac8 = ac8 + 1
                elif (i == "A") and (j == "G"):
                    ag8 = ag8 + 1
                elif (i == "A") and (j == "T"):
                    at8 = at8 + 1
                elif (i == "C") and (j == "A"):
                    ca8 = ca8 + 1
                elif (i == "C") and (j == "G"):
                    cg8 = cg8 + 1
                elif (i == "C") and (j == "T"):
                    ct8 = ct8 + 1
                elif (i == "G") and (j == "A"):
                    ga8 = ga8 + 1
                elif (i == "G") and (j == "C"):
                    gc8 = gc8 + 1
                elif (i == "G") and (j == "T"):
                    gt8 = gt8 + 1
                elif (i == "T") and (j == "A"):
                    ta8 = ta8 + 1
                elif (i == "T") and (j == "C"):
                    tc8 = tc8 + 1
                elif (i == "T") and (j == "G"):
                    tg8 = tg8 + 1

            list.extend([aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8,
                         cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8,
                         gg0, gg1, gg2, gg3, gg4, gg5, gg6, gg7, gg8,
                         tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8])
            list.extend([ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8,
                         ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8,
                         at0, at1, at2, at3, at4, at5, at6, at7, at8,
                         ca0, ca1, ca2, ca3, ca4, ca5, ca6, ca7, ca8])
            list.extend([cg0, cg1, cg2, cg3, cg4, cg5, cg6, cg7, cg8,
                         ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8,
                         ga0, ga1, ga2, ga3, ga4, ga5, ga6, ga7, ga8,
                         gc0, gc1, gc2, gc3, gc4, gc5, gc6, gc7, gc8])
            list.extend([gt0, gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8,
                         ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8,
                         tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8,
                         tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7, tg8])

            vectors.append(list)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    chart = pd.DataFrame(vectors, columns=['color'] + FEATURE_COLUMNS)
    return (chart)


# =============================================================================
# Prediction: 16-base sequences (normalized, no color labels)
# =============================================================================
def create_feature_vectors2(inFile):
    vectors = []
    with open(inFile, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            list = []

            aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cc0, cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gg0, gg1, gg2, gg3, gg4, gg5, gg6, gg7, gg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ag0, ag1, ag2, ag3, ag4, ag5, ag6, ag7, ag8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            at0, at1, at2, at3, at4, at5, at6, at7, at8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ca0, ca1, ca2, ca3, ca4, ca5, ca6, ca7, ca8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            cg0, cg1, cg2, cg3, cg4, cg5, cg6, cg7, cg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ct0, ct1, ct2, ct3, ct4, ct5, ct6, ct7, ct8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ga0, ga1, ga2, ga3, ga4, ga5, ga6, ga7, ga8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gc0, gc1, gc2, gc3, gc4, gc5, gc6, gc7, gc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            gt0, gt1, gt2, gt3, gt4, gt5, gt6, gt7, gt8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            ta0, ta1, ta2, ta3, ta4, ta5, ta6, ta7, ta8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tc0, tc1, tc2, tc3, tc4, tc5, tc6, tc7, tc8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            tg0, tg1, tg2, tg3, tg4, tg5, tg6, tg7, tg8 = 0, 0, 0, 0, 0, 0, 0, 0, 0
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            g0, g1, g2, g3, g4, g5, g6, g7, g8, g9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            words = line.split(" ")

            # Distance 1
            for i, j in zip(words[::], words[1::]):
                if (i == "A") and (j == "A"):
                    aa0 = (aa0 + 1)
                elif (i == "C") and (j == "C"):
                    cc0 = cc0 + 1
                elif (i == "G") and (j == "G"):
                    gg0 = gg0 + 1
                elif (i == "T") and (j == "T"):
                    tt0 = tt0 + 1
                elif (i == "A") and (j == "C"):
                    ac0 = ac0 + 1
                elif (i == "A") and (j == "G"):
                    ag0 = ag0 + 1
                elif (i == "A") and (j == "T"):
                    at0 = at0 + 1
                elif (i == "C") and (j == "A"):
                    ca0 = ca0 + 1
                elif (i == "C") and (j == "G"):
                    cg0 = cg0 + 1
                elif (i == "C") and (j == "T"):
                    ct0 = ct0 + 1
                elif (i == "G") and (j == "A"):
                    ga0 = ga0 + 1
                elif (i == "G") and (j == "C"):
                    gc0 = gc0 + 1
                elif (i == "G") and (j == "T"):
                    gt0 = gt0 + 1
                elif (i == "T") and (j == "A"):
                    ta0 = ta0 + 1
                elif (i == "T") and (j == "C"):
                    tc0 = tc0 + 1
                elif (i == "T") and (j == "G"):
                    tg0 = tg0 + 1

            # Distance 2
            for i, j in zip(words[::], words[2::]):
                if (i == "A") and (j == "A"):
                    aa1 = aa1 + 1
                elif (i == "C") and (j == "C"):
                    cc1 = cc1 + 1
                elif (i == "G") and (j == "G"):
                    gg1 = gg1 + 1
                elif (i == "T") and (j == "T"):
                    tt1 = tt1 + 1
                elif (i == "A") and (j == "C"):
                    ac1 = ac1 + 1
                elif (i == "A") and (j == "G"):
                    ag1 = ag1 + 1
                elif (i == "A") and (j == "T"):
                    at1 = at1 + 1
                elif (i == "C") and (j == "A"):
                    ca1 = ca1 + 1
                elif (i == "C") and (j == "G"):
                    cg1 = cg1 + 1
                elif (i == "C") and (j == "T"):
                    ct1 = ct1 + 1
                elif (i == "G") and (j == "A"):
                    ga1 = ga1 + 1
                elif (i == "G") and (j == "C"):
                    gc1 = gc1 + 1
                elif (i == "G") and (j == "T"):
                    gt1 = gt1 + 1
                elif (i == "T") and (j == "A"):
                    ta1 = ta1 + 1
                elif (i == "T") and (j == "C"):
                    tc1 = tc1 + 1
                elif (i == "T") and (j == "G"):
                    tg1 = tg1 + 1

            # Distance 3
            for i, j in zip(words[::], words[3::]):
                if (i == "A") and (j == "A"):
                    aa2 = aa2 + 1
                elif (i == "C") and (j == "C"):
                    cc2 = cc2 + 1
                elif (i == "G") and (j == "G"):
                    gg2 = gg2 + 1
                elif (i == "T") and (j == "T"):
                    tt2 = tt2 + 1
                elif (i == "A") and (j == "C"):
                    ac2 = ac2 + 1
                elif (i == "A") and (j == "G"):
                    ag2 = ag2 + 1
                elif (i == "A") and (j == "T"):
                    at2 = at2 + 1
                elif (i == "C") and (j == "A"):
                    ca2 = ca2 + 1
                elif (i == "C") and (j == "G"):
                    cg2 = cg2 + 1
                elif (i == "C") and (j == "T"):
                    ct2 = ct2 + 1
                elif (i == "G") and (j == "A"):
                    ga2 = ga2 + 1
                elif (i == "G") and (j == "C"):
                    gc2 = gc2 + 1
                elif (i == "G") and (j == "T"):
                    gt2 = gt2 + 1
                elif (i == "T") and (j == "A"):
                    ta2 = ta2 + 1
                elif (i == "T") and (j == "C"):
                    tc2 = tc2 + 1
                elif (i == "T") and (j == "G"):
                    tg2 = tg2 + 1

            # Distance 4
            for i, j in zip(words[::], words[4::]):
                if (i == "A") and (j == "A"):
                    aa3 = aa3 + 1
                elif (i == "C") and (j == "C"):
                    cc3 = cc3 + 1
                elif (i == "G") and (j == "G"):
                    gg3 = gg3 + 1
                elif (i == "T") and (j == "T"):
                    tt3 = tt3 + 1
                elif (i == "A") and (j == "C"):
                    ac3 = ac3 + 1
                elif (i == "A") and (j == "G"):
                    ag3 = ag3 + 1
                elif (i == "A") and (j == "T"):
                    at3 = at3 + 1
                elif (i == "C") and (j == "A"):
                    ca3 = ca3 + 1
                elif (i == "C") and (j == "G"):
                    cg3 = cg3 + 1
                elif (i == "C") and (j == "T"):
                    ct3 = ct3 + 1
                elif (i == "G") and (j == "A"):
                    ga3 = ga3 + 1
                elif (i == "G") and (j == "C"):
                    gc3 = gc3 + 1
                elif (i == "G") and (j == "T"):
                    gt3 = gt3 + 1
                elif (i == "T") and (j == "A"):
                    ta3 = ta3 + 1
                elif (i == "T") and (j == "C"):
                    tc3 = tc3 + 1
                elif (i == "T") and (j == "G"):
                    tg3 = tg3 + 1

            # Distance 5
            for i, j in zip(words[::], words[5::]):
                if (i == "A") and (j == "A"):
                    aa4 = aa4 + 1
                elif (i == "C") and (j == "C"):
                    cc4 = cc4 + 1
                elif (i == "G") and (j == "G"):
                    gg4 = gg4 + 1
                elif (i == "T") and (j == "T"):
                    tt4 = tt4 + 1
                elif (i == "A") and (j == "C"):
                    ac4 = ac4 + 1
                elif (i == "A") and (j == "G"):
                    ag4 = ag4 + 1
                elif (i == "A") and (j == "T"):
                    at4 = at4 + 1
                elif (i == "C") and (j == "A"):
                    ca4 = ca4 + 1
                elif (i == "C") and (j == "G"):
                    cg4 = cg4 + 1
                elif (i == "C") and (j == "T"):
                    ct4 = ct4 + 1
                elif (i == "G") and (j == "A"):
                    ga4 = ga4 + 1
                elif (i == "G") and (j == "C"):
                    gc4 = gc4 + 1
                elif (i == "G") and (j == "T"):
                    gt4 = gt4 + 1
                elif (i == "T") and (j == "A"):
                    ta4 = ta4 + 1
                elif (i == "T") and (j == "C"):
                    tc4 = tc4 + 1
                elif (i == "T") and (j == "G"):
                    tg4 = tg4 + 1

            # Distance 6
            for i, j in zip(words[::], words[6::]):
                if (i == "A") and (j == "A"):
                    aa5 = aa5 + 1
                elif (i == "C") and (j == "C"):
                    cc5 = cc5 + 1
                elif (i == "G") and (j == "G"):
                    gg5 = gg5 + 1
                elif (i == "T") and (j == "T"):
                    tt5 = tt5 + 1
                elif (i == "A") and (j == "C"):
                    ac5 = ac5 + 1
                elif (i == "A") and (j == "G"):
                    ag5 = ag5 + 1
                elif (i == "A") and (j == "T"):
                    at5 = at5 + 1
                elif (i == "C") and (j == "A"):
                    ca5 = ca5 + 1
                elif (i == "C") and (j == "G"):
                    cg5 = cg5 + 1
                elif (i == "C") and (j == "T"):
                    ct5 = ct5 + 1
                elif (i == "G") and (j == "A"):
                    ga5 = ga5 + 1
                elif (i == "G") and (j == "C"):
                    gc5 = gc5 + 1
                elif (i == "G") and (j == "T"):
                    gt5 = gt5 + 1
                elif (i == "T") and (j == "A"):
                    ta5 = ta5 + 1
                elif (i == "T") and (j == "C"):
                    tc5 = tc5 + 1
                elif (i == "T") and (j == "G"):
                    tg5 = tg5 + 1

            # Distance 7
            for i, j in zip(words[::], words[7::]):
                if (i == "A") and (j == "A"):
                    aa6 = aa6 + 1
                elif (i == "C") and (j == "C"):
                    cc6 = cc6 + 1
                elif (i == "G") and (j == "G"):
                    gg6 = gg6 + 1
                elif (i == "T") and (j == "T"):
                    tt6 = tt6 + 1
                elif (i == "A") and (j == "C"):
                    ac6 = ac6 + 1
                elif (i == "A") and (j == "G"):
                    ag6 = ag6 + 1
                elif (i == "A") and (j == "T"):
                    at6 = at6 + 1
                elif (i == "C") and (j == "A"):
                    ca6 = ca6 + 1
                elif (i == "C") and (j == "G"):
                    cg6 = cg6 + 1
                elif (i == "C") and (j == "T"):
                    ct6 = ct6 + 1
                elif (i == "G") and (j == "A"):
                    ga6 = ga6 + 1
                elif (i == "G") and (j == "C"):
                    gc6 = gc6 + 1
                elif (i == "G") and (j == "T"):
                    gt6 = gt6 + 1
                elif (i == "T") and (j == "A"):
                    ta6 = ta6 + 1
                elif (i == "T") and (j == "C"):
                    tc6 = tc6 + 1
                elif (i == "T") and (j == "G"):
                    tg6 = tg6 + 1

            # Distance 8
            for i, j in zip(words[::], words[8::]):
                if (i == "A") and (j == "A"):
                    aa7 = aa7 + 1
                elif (i == "C") and (j == "C"):
                    cc7 = cc7 + 1
                elif (i == "G") and (j == "G"):
                    gg7 = gg7 + 1
                elif (i == "T") and (j == "T"):
                    tt7 = tt7 + 1
                elif (i == "A") and (j == "C"):
                    ac7 = ac7 + 1
                elif (i == "A") and (j == "G"):
                    ag7 = ag7 + 1
                elif (i == "A") and (j == "T"):
                    at7 = at7 + 1
                elif (i == "C") and (j == "A"):
                    ca7 = ca7 + 1
                elif (i == "C") and (j == "G"):
                    cg7 = cg7 + 1
                elif (i == "C") and (j == "T"):
                    ct7 = ct7 + 1
                elif (i == "G") and (j == "A"):
                    ga7 = ga7 + 1
                elif (i == "G") and (j == "C"):
                    gc7 = gc7 + 1
                elif (i == "G") and (j == "T"):
                    gt7 = gt7 + 1
                elif (i == "T") and (j == "A"):
                    ta7 = ta7 + 1
                elif (i == "T") and (j == "C"):
                    tc7 = tc7 + 1
                elif (i == "T") and (j == "G"):
                    tg7 = tg7 + 1

            # Distance 9
            for i, j in zip(words[::], words[9::]):
                if (i == "A") and (j == "A"):
                    aa8 = aa8 + 1
                elif (i == "C") and (j == "C"):
                    cc8 = cc8 + 1
                elif (i == "G") and (j == "G"):
                    gg8 = gg8 + 1
                elif (i == "T") and (j == "T"):
                    tt8 = tt8 + 1
                elif (i == "A") and (j == "C"):
                    ac8 = ac8 + 1
                elif (i == "A") and (j == "G"):
                    ag8 = ag8 + 1
                elif (i == "A") and (j == "T"):
                    at8 = at8 + 1
                elif (i == "C") and (j == "A"):
                    ca8 = ca8 + 1
                elif (i == "C") and (j == "G"):
                    cg8 = cg8 + 1
                elif (i == "C") and (j == "T"):
                    ct8 = ct8 + 1
                elif (i == "G") and (j == "A"):
                    ga8 = ga8 + 1
                elif (i == "G") and (j == "C"):
                    gc8 = gc8 + 1
                elif (i == "G") and (j == "T"):
                    gt8 = gt8 + 1
                elif (i == "T") and (j == "A"):
                    ta8 = ta8 + 1
                elif (i == "T") and (j == "C"):
                    tc8 = tc8 + 1
                elif (i == "T") and (j == "G"):
                    tg8 = tg8 + 1

            # Apply normalization weights for 16-base sequences
            list.extend([aa0 * (9 / 15), aa1 * (8 / 14), aa2 * (7 / 13), aa3 * (6 / 12), aa4 * (5 / 11), aa5 * (4 / 10), aa6 * (3 / 9), aa7 * (2 / 8), aa8 * (1 / 7),
                         cc0 * (9 / 15), cc1 * (8 / 14), cc2 * (7 / 13), cc3 * (6 / 12), cc4 * (5 / 11), cc5 * (4 / 10), cc6 * (3 / 9), cc7 * (2 / 8), cc8 * (1 / 7),
                         gg0 * (9 / 15), gg1 * (8 / 14), gg2 * (7 / 13), gg3 * (6 / 12), gg4 * (5 / 11), gg5 * (4 / 10), gg6 * (3 / 9), gg7 * (2 / 8), gg8 * (1 / 7),
                         tt0 * (9 / 15), tt1 * (8 / 14), tt2 * (7 / 13), tt3 * (6 / 12), tt4 * (5 / 11), tt5 * (4 / 10), tt6 * (3 / 9), tt7 * (2 / 8), tt8 * (1 / 7)])
            list.extend([ac0 * (9 / 15), ac1 * (8 / 14), ac2 * (7 / 13), ac3 * (6 / 12), ac4 * (5 / 11), ac5 * (4 / 10), ac6 * (3 / 9), ac7 * (2 / 8), ac8 * (1 / 7),
                         ag0 * (9 / 15), ag1 * (8 / 14), ag2 * (7 / 13), ag3 * (6 / 12), ag4 * (5 / 11), ag5 * (4 / 10), ag6 * (3 / 9), ag7 * (2 / 8), ag8 * (1 / 7),
                         at0 * (9 / 15), at1 * (8 / 14), at2 * (7 / 13), at3 * (6 / 12), at4 * (5 / 11), at5 * (4 / 10), at6 * (3 / 9), at7 * (2 / 8), at8 * (1 / 7),
                         ca0 * (9 / 15), ca1 * (8 / 14), ca2 * (7 / 13), ca3 * (6 / 12), ca4 * (5 / 11), ca5 * (4 / 10), ca6 * (3 / 9), ca7 * (2 / 8), ca8 * (1 / 7)])
            list.extend([cg0 * (9 / 15), cg1 * (8 / 14), cg2 * (7 / 13), cg3 * (6 / 12), cg4 * (5 / 11), cg5 * (4 / 10), cg6 * (3 / 9), cg7 * (2 / 8), cg8 * (1 / 7),
                         ct0 * (9 / 15), ct1 * (8 / 14), ct2 * (7 / 13), ct3 * (6 / 12), ct4 * (5 / 11), ct5 * (4 / 10), ct6 * (3 / 9), ct7 * (2 / 8), ct8 * (1 / 7),
                         ga0 * (9 / 15), ga1 * (8 / 14), ga2 * (7 / 13), ga3 * (6 / 12), ga4 * (5 / 11), ga5 * (4 / 10), ga6 * (3 / 9), ga7 * (2 / 8), ga8 * (1 / 7),
                         gc0 * (9 / 15), gc1 * (8 / 14), gc2 * (7 / 13), gc3 * (6 / 12), gc4 * (5 / 11), gc5 * (4 / 10), gc6 * (3 / 9), gc7 * (2 / 8), gc8 * (1 / 7)])
            list.extend([gt0 * (9 / 15), gt1 * (8 / 14), gt2 * (7 / 13), gt3 * (6 / 12), gt4 * (5 / 11), gt5 * (4 / 10), gt6 * (3 / 9), gt7 * (2 / 8), gt8 * (1 / 7),
                         ta0 * (9 / 15), ta1 * (8 / 14), ta2 * (7 / 13), ta3 * (6 / 12), ta4 * (5 / 11), ta5 * (4 / 10), ta6 * (3 / 9), ta7 * (2 / 8), ta8 * (1 / 7),
                         tc0 * (9 / 15), tc1 * (8 / 14), tc2 * (7 / 13), tc3 * (6 / 12), tc4 * (5 / 11), tc5 * (4 / 10), tc6 * (3 / 9), tc7 * (2 / 8), tc8 * (1 / 7),
                         tg0 * (9 / 15), tg1 * (8 / 14), tg2 * (7 / 13), tg3 * (6 / 12), tg4 * (5 / 11), tg5 * (4 / 10), tg6 * (3 / 9), tg7 * (2 / 8), tg8 * (1 / 7)])
            vectors.append(list)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    chart = pd.DataFrame(vectors, columns=FEATURE_COLUMNS)
    return (chart)
