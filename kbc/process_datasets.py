# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import errno
import shutil
import pickle

import numpy as np

from collections import defaultdict


def prepare_dataset(path):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    out_path = os.path.join(path, 'kbc_data')
    files = ['train.txt', 'valid.txt', 'test.txt']

    q2b_maps = ['ind2ent.pkl', 'ind2rel.pkl']
    if all([os.path.exists(os.path.join(path, f)) for f in q2b_maps]):
        # Read IDs from q2b mappings
        with open(os.path.join(path, q2b_maps[0]), 'rb') as f:
            ind2ent = pickle.load(f)
            entities_to_id = {ent: i for i, ent in ind2ent.items()}
        with open(os.path.join(path, q2b_maps[1]), 'rb') as f:
            ind2rel = pickle.load(f)
            relations_to_id = {rel: i for i, rel in ind2rel.items()}

        # Create IDs for the remaining entities and relations (not used in q2b)
        max_ent_id = max(entities_to_id.values())
        max_rel_id = max(relations_to_id.values())

        for f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as to_read:
                for line in to_read.readlines():
                    lhs, rel, rhs = line.strip().split('\t')
                    if lhs not in entities_to_id:
                        max_ent_id += 1
                        entities_to_id[lhs] = max_ent_id
                    if rhs not in entities_to_id:
                        max_ent_id += 1
                        entities_to_id[rhs] = max_ent_id
                    if rel not in relations_to_id:
                        max_rel_id += 1
                        relations_to_id[rel] = max_rel_id

    else:
        entities, relations = set(), set()
        for f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as to_read:
                for line in to_read.readlines():
                    lhs, rel, rhs = line.strip().split('\t')
                    print(rel)
                    entities.add(lhs)
                    entities.add(rhs)
                    relations.add(rel)

        entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
        relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}

    n_relations = len(relations_to_id)
    n_entities = len(entities_to_id)
    print(f'{n_entities} entities and {n_relations} relations')

    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    os.makedirs(out_path)
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id], ['ent_id', 'rel_id']):
        pickle.dump(dic, open(os.path.join(out_path, f'{f}.pickle'), 'wb'))

    # map train/test/valid with the ids
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs = line.strip().split('\t')

            lhs_id = entities_to_id[lhs]
            rhs_id = entities_to_id[rhs]
            rel_id = relations_to_id[rel]
            inv_rel_id = relations_to_id[rel + '_reverse']

            examples.append([lhs_id, rel_id, rhs_id])
            to_skip['rhs'][(lhs_id, rel_id)].add(rhs_id)
            to_skip['lhs'][(rhs_id, inv_rel_id)].add(lhs_id)

            # Add inverse relations for training
            if f == 'train.txt':
                examples.append([rhs_id, inv_rel_id, lhs_id])
                to_skip['rhs'][(rhs_id, inv_rel_id)].add(lhs_id)
                to_skip['lhs'][(lhs_id, rel_id)].add(rhs_id)

        out = open(os.path.join(out_path, f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(os.path.join(out_path, 'to_skip.pickle'), 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(os.path.join(out_path, 'train.txt.pickle'), 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(os.path.join(out_path, 'probas.pickle'), 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Relational learning contraption"
    )
    parser.add_argument('data_path', help='Path containing triples for'
                                            ' training, validation, and test')
    args = parser.parse_args()
    data_path = args.data_path

    print(f'Loading dataset from {data_path}')
    try:
        prepare_dataset(data_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print(e)
            print("File exists. skipping...")
        else:
            raise
