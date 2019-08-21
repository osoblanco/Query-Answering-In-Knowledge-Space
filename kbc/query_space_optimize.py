import argparse
import pickle

import torch


import numpy as np

from kbc.learn import kbc_model_load
from kbc.learn import dataset_to_query

from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import Chain


def get_optimization(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess_raw, closest_map = None, None
    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)
        queries,target_ids = dataset_to_query(kbc.model,dataset,dataset_mode)

        print(queries[0].dim())

        print(len(queries[0]))

        lhs_norm = 0.0
        for lhs_emb in queries[0]:
            lhs_norm+=torch.norm(lhs_emb)

        lhs_norm/= len(queries[0])

        obj_guess_raw, closest_map, indices_rankedby_distances = kbc.model.projected_gradient_descent(queries, kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)

        guess_norm = 0.0
        for obj_emb in obj_guess_raw:
            guess_norm+=torch.norm(obj_emb)

        guess_norm/= len(obj_guess_raw)
        print("\n")
        print("The average norm of the trained vectors is {}, while optimized vectors have {}".format(lhs_norm,guess_norm))

        predicted_ids = [x[0] for x in closest_map]

        correct = 0.0

        for i in range(len(predicted_ids)):
            if predicted_ids[i] in target_ids[i]:
                # print(predicted_ids[i],target_ids[i])
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(target_ids))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances)):
            correct_ans_indices = [(indices_rankedby_distances[i]== one_target).nonzero()[0].squeeze() for one_target in target_ids[i]]

            correct_ans_index = min(correct_ans_indices)

            if correct_ans_index >1000:
                correct_ans_index = 1000


            print(correct_ans_index)


            average_percentile_rank += 1.0 - float(correct_ans_index) / 1000

        average_percentile_rank /= len(indices_rankedby_distances)

        print("Average Percentile Ranks is: ", average_percentile_rank)

    except RuntimeError as e:
        print("Cannot Optimise the Query space with error: {}".format(str(e)))
        return None, None

    return obj_guess_raw, closest_map



def get_type12_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess_raw, closest_map = None, None

    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)

        raw = dataset.type1_2chain

        type1_2chain = []
        for i in range(len(raw)):
            type1_2chain.append(raw[i].data)


        part1 = [x['raw_chain'][0] for x in type1_2chain]
        part2 = [x['raw_chain'][1] for x in type1_2chain]


        flattened_part1 =[]
        flattened_part2 = []
        target_ids = []

        # [A,b,C][C,d,[Es]]

        for chain_iter in range(len(part2)):
            for target in part2[chain_iter][2]:
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],target])
                flattened_part1.append(part1[chain_iter])
                target_ids.append


        target_ids = {}

        for chain_iter in range(len(part2)):

            key = [part1[chain_iter][0],part1[chain_iter][1],part2[chain_iter][1],part1[chain_iter][2]]
            key = '_'.join(str(e) for e in key)

            if key not in target_ids:
                target_ids[key] = []

            target_ids[key].append(part1[chain_iter][2])



        part1 = flattened_part1
        part2 = flattened_part2


        part1 = np.array(part1)
        part1 = torch.from_numpy(part1.astype('int64')).cuda()

        part2 = np.array(part2)
        part2 = torch.from_numpy(part2.astype('int64')).cuda()


        chain1 = kbc.model.get_full_embeddigns(part1)
        chain2 = kbc.model.get_full_embeddigns(part2)


        print(len(chain1[0]))

        lhs_norm = 0.0
        for lhs_emb in chain1[0]:
            lhs_norm+=torch.norm(lhs_emb)

        lhs_norm/= len(chain1[0])

        obj_guess_raw, closest_map, indices_rankedby_distances \
        = kbc.model.type1_2chain_optimize(chain1,chain2, kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)


        guess_norm = 0.0
        for obj_emb in obj_guess_raw:
            guess_norm+=torch.norm(obj_emb)

        guess_norm/= len(obj_guess_raw)
        print("\n")
        print("The average norm of the trained vectors is {}, while optimized vectors have {}".format(lhs_norm,guess_norm))

        predicted_ids = [x[0] for x in closest_map]

        correct = 0.0

        for i in range(len(predicted_ids)):

            key = [part1[i][0],part1[i][1],part2[i][1],part1[i][2]]
            key = '_'.join(str(e.item()) for e in key)

            if predicted_ids[i] in target_ids[key]:
                # print(predicted_ids[i],target_ids[i])
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(predicted_ids))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],part2[i][1],part1[i][2]]
            key = '_'.join(str(e.item()) for e in key)
            targets = target_ids[key]

            correct_ans_indices = [(indices_rankedby_distances[i] == one_target).nonzero()[0].squeeze() for one_target in targets]

            correct_ans_index = min(correct_ans_indices)

            if correct_ans_index >1000:
                correct_ans_index = 1000

            average_percentile_rank += 1.0 - float(correct_ans_index) / 1000

        average_percentile_rank /= len(indices_rankedby_distances)

        print("Average Percentile Ranks is: ", average_percentile_rank)


    except RuntimeError as e:
        print(e)
        return None
    return obj_guess_raw, closest_map

if __name__ == "__main__":

    big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    datasets = big_datasets
    dataset_modes = ['valid', 'test', 'train']
    similarity_metrics = ['l2', 'Eculidian', 'cosine']

    parser = argparse.ArgumentParser(
    description="Query space optimizer namespace"
    )

    parser.add_argument(
    '--model_path',
    help="The path to the KBC model. Can be both relative and full"
    )

    parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
    )

    parser.add_argument(
    '--dataset_mode', choices=dataset_modes, default='train',
    help="Dataset validation mode in {}".format(dataset_modes)
    )
    parser.add_argument(
    '--similarity_metric', choices=similarity_metrics, default='l2',
    help="Dataset validation mode in {}".format(similarity_metrics)
    )

    args = parser.parse_args()

    # obj_guess, closest_map =  get_optimization(args.model_path, args.dataset, args.dataset_mode, args.similarity_metric)


    WN = pickle.load(open("Bio.pkl",'rb'))

    obj_guess, closest_map =  get_type12_graph_optimizaton(args.model_path, WN, args.dataset_mode, args.similarity_metric)
