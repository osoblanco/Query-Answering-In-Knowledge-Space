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
    obj_guess, closest_map = None, None

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

        # [A,b,C][C,d,[Es]]

        for chain_iter in range(len(part2)):
            for target in part2[chain_iter][2]:
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],target])
                flattened_part1.append(part1[chain_iter])


        part1 = flattened_part1
        part2 = flattened_part2

        # SAMPLING HACK
        if len(part1) > 5000:
            part1 = part1[:5000]
            part2 = part2[:5000]

        target_ids = {}

        for chain_iter in range(len(part2)):

            key = [part1[chain_iter][0],part1[chain_iter][1],part2[chain_iter][1],part2[chain_iter][2]]
            key = '_'.join(str(e) for e in key)

            if key not in target_ids:
                target_ids[key] = []

            target_ids[key].append(part1[chain_iter][2])


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

            key = [part1[i][0],part1[i][1],part2[i][1],part2[i][2]]
            key = '_'.join(str(e.item()) for e in key)

            if predicted_ids[i] in target_ids[key]:
                # print(predicted_ids[i],target_ids[i])
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(predicted_ids))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],part2[i][1],part2[i][2]]
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

def get_type22_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess_raw, closest_map = None, None

    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)

        raw = dataset.type2_2chain

        type2_2chain = []
        for i in range(len(raw)):
            type2_2chain.append(raw[i].data)


        part1 = [x['raw_chain'][0] for x in type2_2chain]
        part2 = [x['raw_chain'][1] for x in type2_2chain]


        target_ids = {}

        for chain_iter in range(len(part2)):

            key = [part1[chain_iter][0],part1[chain_iter][1],part2[chain_iter][0],part1[chain_iter][1]]
            key = '_'.join(str(e) for e in key)

            if key not in target_ids:
                target_ids[key] = []

            target_ids[key].append(part1[chain_iter][2])


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
        = kbc.model.type2_2chain_optimize(chain1,chain2, kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)


        guess_norm = 0.0
        for obj_emb in obj_guess_raw:
            guess_norm+=torch.norm(obj_emb)

        guess_norm/= len(obj_guess_raw)
        print("\n")
        print("The average norm of the trained vectors is {}, while optimized vectors have {}".format(lhs_norm,guess_norm))

        predicted_ids = [x[0] for x in closest_map]

        correct = 0.0

        for i in range(len(predicted_ids)):

            key = [part1[i][0],part1[i][1],part2[i][0],part1[i][1]]
            key = '_'.join(str(e.item()) for e in key)

            if predicted_ids[i] in target_ids[key]:
                # print(predicted_ids[i],target_ids[i])
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(predicted_ids))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],part2[i][0],part1[i][1]]
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



def get_type13_graph_optimizaton_joint(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):

    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)

        raw = dataset.type1_3chain

        type1_3chain = []
        for i in range(len(raw)):
            type1_3chain.append(raw[i].data)


        part1 = [x['raw_chain'][0] for x in type1_3chain]
        part2 = [x['raw_chain'][1] for x in type1_3chain]
        part3 = [x['raw_chain'][2] for x in type1_3chain]


        flattened_part1 =[]
        flattened_part2 = []
        flattened_part3 = []

        # [A,b,C][C,d,[Es]]

        for chain_iter in range(len(part3)):
            for target in part3[chain_iter][2]:

                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],target])
                flattened_part2.append(part2[chain_iter])
                flattened_part1.append(part1[chain_iter])


        part1 = flattened_part1
        part2 = flattened_part2
        part3 = flattened_part3


        # SAMPLING HACK
        if len(part1) > 8000:
            part1 = part1[:8000]
            part2 = part2[:8000]
            part3 = part3[:8000]

        target_ids = {}

        for chain_iter in range(len(part1)):

            key = [part1[chain_iter][0],part1[chain_iter][1],\
                    part2[chain_iter][1],\
                        part3[chain_iter][1],part3[chain_iter][2]
                ]

            key = '_'.join(str(e) for e in key)

            if key not in target_ids:
                target_ids[key] = []

            target_ids[key].append((part1[chain_iter][2],part2[chain_iter][2]))

        part1 = np.array(part1)
        part1 = torch.from_numpy(part1.astype('int64')).cuda()

        part2 = np.array(part2)
        part2 = torch.from_numpy(part2.astype('int64')).cuda()

        part3 = np.array(part3)
        part3 = torch.from_numpy(part3.astype('int64')).cuda()

        chain1 = kbc.model.get_full_embeddigns(part1)
        chain2 = kbc.model.get_full_embeddigns(part2)
        chain3 = kbc.model.get_full_embeddigns(part3)


        print(len(chain1[0]))

        lhs_norm = 0.0
        for lhs_emb in chain1[0]:
            lhs_norm+=torch.norm(lhs_emb)

        lhs_norm/= len(chain1[0])

        obj_guess_raw_1, obj_guess_raw_2, closest_map_1,closest_map_2, \
        indices_rankedby_distances_1, indices_rankedby_distances_2 \
        = kbc.model.type1_3chain_optimize_joint(chain1,chain2,chain3, kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)


        guess_norm_1 = 0.0
        for obj_emb in obj_guess_raw_1:
            guess_norm_1 +=torch.norm(obj_emb)

        guess_norm_1 /= len(obj_guess_raw_1)

        guess_norm_2 = 0.0
        for obj_emb in obj_guess_raw_2:
            guess_norm_2 +=torch.norm(obj_emb)

        guess_norm_2 /= len(obj_guess_raw_2)

        guess_norm = (guess_norm_1 + guess_norm_2)/2.0

        print("\n")
        print("The average norm of the trained vectors is {}, while optimized vectors have {}".format(lhs_norm,guess_norm))

        predicted_ids_1 = [x[0] for x in closest_map_1]
        predicted_ids_2 = [x[0] for x in closest_map_2]

        correct = 0.0

        for i in range(len(predicted_ids_1)):

            key = [part1[i][0],part1[i][1],\
                    part2[i][1],\
                        part3[i][1],part3[i][2]
                ]

            key = '_'.join(str(e.item()) for e in key)

            if (int(predicted_ids_1[i].item()),int(predicted_ids_2[i].item())) in target_ids[key]:
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(predicted_ids_1))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances_1)):

            key = [part1[i][0],part1[i][1],\
                    part2[i][1],\
                        part3[i][1],part3[i][2]
                ]

            key = '_'.join(str(e.item()) for e in key)
            targets = target_ids[key]

            correct_ans_indices_1 = [(indices_rankedby_distances_1[i] == one_target[0]).nonzero()[0].squeeze() for one_target in targets]
            correct_ans_indices_2 = [(indices_rankedby_distances_2[i] == one_target[1]).nonzero()[0].squeeze() for one_target in targets]

            correct_ans_index_1 = min(correct_ans_indices_1)
            correct_ans_index_2 = min(correct_ans_indices_2)

            correct_ans_index = (correct_ans_index_1 + correct_ans_index_2)/2.0

            if correct_ans_index >1000:
                correct_ans_index = 1000

            average_percentile_rank += 1.0 - float(correct_ans_index) / 1000

        average_percentile_rank /= len(indices_rankedby_distances_1)

        print("Average Percentile Ranks is: ", average_percentile_rank)


    except RuntimeError as e:
        print(e)
        return None
    return "Completed"




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

    ans =  get_type13_graph_optimizaton_joint(args.model_path, WN, args.dataset_mode, args.similarity_metric)
