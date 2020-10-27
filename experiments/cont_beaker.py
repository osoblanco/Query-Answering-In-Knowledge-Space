#!/usr/bin/env python3
# -*- coding: utf-8 -*-

dsets = ['FB15k', 'FB15k-237', 'NELL']
ranks = [100, 200, 500, 1000]
tnorms = ['min', 'product']
weights = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

chain_types = ['1_2','1_3', '2_2', '2_3', '3_3', '4_3', '2_2_disj', '4_3_disj']

cmd_lines = []

for d in dsets:
    for r in ranks:
        for t in tnorms:
            for w in weights:
                for dev in ['valid', 'test']:
                    for q in chain_types:

                        m = d

                        cmd = f"PYTHONPATH=. python3 kbc/continuous.py --model_path models/{m}-model-rank-{r}-epoch-100-*.pt --dataset {d} --dataset_mode {dev} --chain_type {q} --reg {w}"
                        
                        _id = f"cont_d={d}_r={r}_t={t}_w={w}_dev={dev}_q={q}"

                        cmd_lines += [f"{cmd} > logs/cont/cont_{_id}.log 2>&1"]

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
