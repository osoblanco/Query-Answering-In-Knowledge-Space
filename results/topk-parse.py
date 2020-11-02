#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import os.path

import json

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def path_to_key(path):
    res = path.split('.')[0].split('/')[-1].replace('topk_', '')
    return res


def path_to_results(path):
    with open(path) as f:
        res = json.load(f)
    return res


def main(argv):
    key_to_path = {path_to_key(path): path for path in argv}

    key_lst = sorted([key for key in key_to_path])

    for d in ['FB15K', 'FB237', 'NELL']:
        for rank in [100, 200, 500, 1000]:
            results = []

            # for query in ['1_2', '1_3', '2_2', '2_3', '3_3', '4_3', '2_2_disj', '4_3_disj']:
            for query in ['1_2', '1_3', '2_2', '2_3', '4_3', '3_3', '2_2_disj', '4_3_disj']:

                _keys = []
                for key in key_lst:
                    # print(d, rank, query, key)
                    if f'd={d}_dev' in key and f'rank={rank}_' in key and f'e={query}_' in key:
                        _keys += [key]

                best_value = None
                best_dev_key = None

                for key in _keys:
                    res = path_to_results(key_to_path[key])
                    if best_value is None or res["HITS@3m_new"] > best_value:
                        _tmp = key.replace('_dev', '')                        
                        if _tmp in key_to_path and os.path.isfile(key_to_path[_tmp]): 
                            best_dev_key = key
                            best_value = res["HITS@3m_new"]

                best_test_key = best_dev_key.replace('_dev', '')
                best_test_path = key_to_path[best_test_key]

                #print(best_test_path)
                #if '2_2_rank' in best_test_path:
                #    print(best_test_path)

                res = path_to_results(best_test_path)
                results += [res["HITS@3m_new"]]

            print(f'd={d} rank={rank} & ' + " & ".join([f'{r:.3f}' for r in results]))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
