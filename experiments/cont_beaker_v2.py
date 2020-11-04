#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random

dsets = ['FB15k', 'FB15k-237', 'NELL']
ranks = [100, 200, 500, 1000]
tnorms = ['min', 'product']
weights = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]

lrs = [0.1, 0.01, 0.001]
optimizers = ['adam', 'adagrad']
max_steps_lst = [0, 100, 1000]

chain_types = ['1_2','1_3', '2_2', '2_3', '3_3', '4_3', '2_2_disj', '4_3_disj']

cmd_lines = []

for d in dsets:
    for r in ranks:
        for t in tnorms:
            for w in weights:
                for dev in ['valid', 'test']:
                    for lr in lrs:
                        for o in optimizers:
                            for ms in max_steps_lst:
                                for q in chain_types:

                                    m = d

                                    cmd = f"PYTHONPATH=. python3 kbc/continuous.py --model_path models/{m}-model-rank-{r}-epoch-100-*.pt " \
                                          f"--dataset {d} --dataset_mode {dev} --chain_type {q} --reg {w} --lr {lr} --optimizer {o} --max-steps {ms} "

                                    _id = f"cont_d={d}_r={r}_t={t}_w={w}_dev={dev}_q={q}_lr={lr}_o={o}_ms={ms}"

                                    cmd_lines += [f"{cmd} > logs/cont/cont_{_id}.log 2>&1"]

random.Random(0).shuffle(cmd_lines)

nb_jobs = len(cmd_lines)

header = """#!/bin/bash -l

#$ -cwd
#$ -S /bin/bash
#$ -o $HOME/array.out
#$ -e $HOME/array.err
#$ -t 1-{}
#$ -l tmem=12G
#$ -l h_rt=2:00:00
#$ -l gpu=true

conda activate gpu

export LANG="en_US.utf8"
export LANGUAGE="en_US:en"

cd $HOME/workspace/Query-Answering-In-Knowledge-Space

""".format(nb_jobs)

print(header)

for job_id, command_line in enumerate(cmd_lines, 1):
    print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))
