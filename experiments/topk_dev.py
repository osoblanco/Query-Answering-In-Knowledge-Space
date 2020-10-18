#!/usr/bin/env python3
# -*- coding: utf-8 -*-

dsets = ['FB15K', 'FB237', 'NELL']
ranks = [100, 200, 500, 1000]
tnorms = ['min', 'product']
candidates = [3, 4, 5]

for d in dsets:
    for r in ranks:
        for t in tnorms:
            for c in candidates:

                m = d
                if d == 'FB237':
                    m = "FB15k-237"
                elif d == 'FB15K':
                    m = 'FB15k'

                cmd = f"PYTHONPATH=. python3 kbc/query_answer_BF.py --model_path models/{m}-model-rank-{r}-epoch-100-*.pt " \
                    f"--dataset {d}_dev --dataset_mode test --t_norm {t} --candidates {c}"

                log_path = f"logs/topk_dev_d={d}_r={r}_t={t}_c={c}.log"
                print(f"{cmd} > {log_path} 2>&1")
