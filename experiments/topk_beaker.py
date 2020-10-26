#!/usr/bin/env python3
# -*- coding: utf-8 -*-

dsets = ['FB15K', 'FB237', 'NELL']
ranks = [100, 200, 500, 1000]
tnorms = ['min', 'product']
candidates = [3, 4, 5, 6]

cmd_lines = []

for d in dsets:
    for r in ranks:
        for t in tnorms:
            for c in candidates:
                for is_dev in [True, False]:

                    m = d
                    if d == 'FB237':
                        m = "FB15k-237"
                    elif d == 'FB15K':
                        m = 'FB15k'

                    _d = d if is_dev is False else f'{d}_dev'

                    cmd = f"PYTHONPATH=. python3 kbc/query_answer_BF.py --model_path models/{m}-model-rank-{r}-epoch-100-*.pt " \
                        f"--dataset {_d} --dataset_mode test --t_norm {t} --candidates {c}"

                    _id = f"topk_d={_d}_r={r}_t={t}_c={c}"

                    cmd_lines += [f"{cmd} > logs/topk/topk_{_id}.log 2>&1"]

nb_jobs = len(cmd_lines)

header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=16G
#$ -l h_rt=4:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/Query-Answering-In-Knowledge-Space

""".format(nb_jobs)

print(header)

for job_id, command_line in enumerate(cmd_lines, 1):
    print('test $SGE_TASK_ID -eq {} && sleep 30 && {}'.format(job_id, command_line))
