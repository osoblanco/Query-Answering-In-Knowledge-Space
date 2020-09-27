import argparse
import pickle

import torch


import numpy as np

from kbc.exhaustive_objective_search import exhaustive_objective_search


from kbc.learn import kbc_model_load
from kbc.learn import dataset_to_query

from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import Chain
from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton
from kbc.utils import preload_env


from kbc.metrics import average_percentile_rank
from kbc.metrics import norm_comparison
from kbc.metrics import hits_at_k, evaluation

def get_optimization(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess_raw, closest_map = None, None
    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)
        queries,target_ids = dataset_to_query(kbc.model,dataset,dataset_mode)


        obj_guess_raw, closest_map, indices_rankedby_distances = kbc.model.projected_gradient_descent(queries, \
                                            kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)

        lhs_norm,  guess_norm =  norm_comparison(queries, obj_guess_raw)


        predicted_ids = [x[0] for x in closest_map]

        correct = 0.0
        for i in range(len(predicted_ids)):
            if predicted_ids[i] in target_ids[i]:
                correct+=1.0

        print("Accuracy at {}".format(correct/(len(target_ids))))


        average_percentile_rank = 0.0
        for i in range(len(indices_rankedby_distances)):
            correct_ans_indices = [(indices_rankedby_distances[i]== one_target).nonzero()[0].squeeze() for one_target in target_ids[i]]

            correct_ans_index = min(correct_ans_indices)

            if correct_ans_index >1000:
                correct_ans_index = 1000

            average_percentile_rank += 1.0 - float(correct_ans_index) / 1000

        average_percentile_rank /= len(indices_rankedby_distances)

        print("Average Percentile Ranks is: ", average_percentile_rank)

    except RuntimeError as e:
        print("Cannot Optimise the Query space with error: {}".format(str(e)))
        return None, None

    return obj_guess_raw, closest_map

def optimize_chains(query_type, kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min'):
    try:
        env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
        env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete

        scores = kbc.model.optimize_chains(chains, kbc.regularizer, max_steps=1000, similarity_metric=similarity_metric, t_norm=t_norm)

        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


def optimize_intersections(query_type, kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min'):
    try:
        env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
        env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete

        scores = kbc.model.optimize_intersections(chains, kbc.regularizer, max_steps=1000, similarity_metric=similarity_metric, t_norm=t_norm)

        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


def get_type13_graph_optimizaton_joint(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm = 'min'):

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

        # [A,b,C][C,d,[E's]]

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



        lhs_norm = 0.0
        for lhs_emb in chain1[0]:
            lhs_norm+=torch.norm(lhs_emb)

        lhs_norm/= len(chain1[0])

        obj_guess_raw_1, obj_guess_raw_2, closest_map_1,closest_map_2, \
        indices_rankedby_distances_1, indices_rankedby_distances_2 \
        = kbc.model.type1_3chain_optimize_joint(chain1,chain2,chain3, kbc.regularizer,\
        max_steps=1000,similarity_metric=similarity_metric, t_norm = t_norm)


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


def get_type13_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm = 'min'):

    try:

        env = preload_env(kbc_path, dataset, dataset_mode, '1_3')
        part1, part2, part3 = env.parts
        target_ids,lhs_norm  = env.target_ids, env.lhs_norm
        kbc, chains = env.kbc, env.chains

        obj_guess_raw,closest_map,indices_rankedby_distances \
        = kbc.model.type1_3chain_optimize(chains, kbc.regularizer,\
        max_steps=1000,similarity_metric=similarity_metric, t_norm = t_norm)

        lhs_norm,  guess_norm =  norm_comparison(lhs_norm, obj_guess_raw)

        keys = []
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],\
                    part2[i][0],part2[i][1],\
                        part3[i][1],part3[i][2]
                ]

            key = '_'.join(str(e.item()) for e in key)
            keys.append(key)

        hits = hits_at_k(indices_rankedby_distances, target_ids, keys, hits = [1,3,5,10,20])

        APR = average_percentile_rank(indices_rankedby_distances,target_ids, keys)


    except RuntimeError as e:
        print(e)
        return None
    return obj_guess_raw,closest_map


def get_type33_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm = 'min'):

    try:
        env = preload_env(kbc_path, dataset, dataset_mode, '3_3')
        part1, part2, part3 = env.parts
        target_ids,lhs_norm  = env.target_ids, env.lhs_norm
        kbc, chains = env.kbc, env.chains

        obj_guess_raw,closest_map,indices_rankedby_distances \
        = kbc.model.type3_3chain_optimize(chains, kbc.regularizer,\
        max_steps=1000,similarity_metric=similarity_metric, t_norm = t_norm)


        lhs_norm,  guess_norm =  norm_comparison(lhs_norm, obj_guess_raw)

        keys = []
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],\
                    part2[i][0],part2[i][1],\
                        part3[i][0],part3[i][1]
                ]

            key = '_'.join(str(e.item()) for e in key)
            keys.append(key)

        hits = hits_at_k(indices_rankedby_distances, target_ids, keys, hits = [1,3,5,10,20])

        APR = average_percentile_rank(indices_rankedby_distances,target_ids, keys)

    except RuntimeError as e:
        print(e)
        return None
    return obj_guess_raw,closest_map

def get_type43_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm='min'):

    try:
        env = preload_env(kbc_path, dataset, dataset_mode, '4_3')
        part1, part2, part3 = env.parts
        target_ids,lhs_norm  = env.target_ids, env.lhs_norm
        kbc, chains = env.kbc, env.chains

        obj_guess_raw,closest_map,indices_rankedby_distances \
        = kbc.model.type4_3chain_optimize(chains, kbc.regularizer,\
        max_steps=1000,similarity_metric=similarity_metric, t_norm=t_norm)

        lhs_norm,  guess_norm =  norm_comparison(lhs_norm, obj_guess_raw)

        keys = []
        for i in range(len(indices_rankedby_distances)):

            key = [part1[i][0],part1[i][1],\
                    part2[i][0],part2[i][1],\
                        part3[i][1],part3[i][2]
                ]

            key = '_'.join(str(e.item()) for e in key)
            keys.append(key)

        hits = hits_at_k(indices_rankedby_distances, target_ids, keys, hits = [1,3,5,10,20])

        APR = average_percentile_rank(indices_rankedby_distances,target_ids, keys)

    except RuntimeError as e:
        print(e)
        return None
    return obj_guess_raw,closest_map

def exhaustive_search_comparison(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm = 'min', graph_type = QuerDAG.TYPE1_2.value):
    try:

        if QuerDAG.TYPE1_2.value in graph_type:
            obj_guess_optim,closest_map_optim = optimize_chains(kbc_path, dataset, dataset_mode, similarity_metric ='l2', t_norm='min')
        if QuerDAG.TYPE2_2.value in graph_type:
            obj_guess_optim,closest_map_optim = optimize_intersections(kbc_path, dataset, dataset_mode, similarity_metric ='l2', t_norm='min')
        if QuerDAG.TYPE1_3.value in graph_type:
            obj_guess_optim,closest_map_optim = get_type13_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm='min')
        if QuerDAG.TYPE2_3.value in graph_type:
            obj_guess_optim,closest_map_optim = optimize_intersections(kbc_path, dataset, dataset_mode, similarity_metric ='l2', t_norm='min')
        if QuerDAG.TYPE3_3.value in graph_type:
            obj_guess_optim,closest_map_optim = get_type33_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm='min')
        if QuerDAG.TYPE4_3.value in graph_type:
            obj_guess_optim,closest_map_optim = get_type43_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2', t_norm='min')

        best_candidates = exhaustive_objective_search(t_norm, graph_type )


    except RuntimeError as e:
        print("Exhastive search Completed with error: ",e)
        return None
    return best_candidates

if __name__ == "__main__":

    big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
    datasets = big_datasets
    dataset_modes = ['valid', 'test', 'train']
    similarity_metrics = ['l2', 'Eculidian', 'cosine']

    chain_types = [QuerDAG.TYPE1_1.value,QuerDAG.TYPE1_2.value,QuerDAG.TYPE2_2.value,QuerDAG.TYPE1_3.value, \
    QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,'All','e']

    t_norms = ['min','product']

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

    parser.add_argument(
    '--chain_type', choices=chain_types, default=QuerDAG.TYPE1_1.value,
    help="Chain type experimenting for ".format(chain_types)
    )

    parser.add_argument(
    '--t_norm', choices=t_norms, default='min',
    help="T-norms available are ".format(t_norms)
    )

    args = parser.parse_args()


    if QuerDAG.TYPE1_1.value in args.chain_type:
        obj_guess, closest_map =  get_optimization(args.model_path, args.dataset, args.dataset_mode, args.similarity_metric)

    else:

        data_hard_path = args.dataset + '_hard.pkl'
        data_complete_path = args.dataset + '_complete.pkl'

        data_hard = pickle.load(open(data_hard_path, 'rb'))
        data_complete = pickle.load(open(data_complete_path, 'rb'))

        if QuerDAG.TYPE1_2.value in args.chain_type:
            ans = optimize_chains(QuerDAG.TYPE1_2.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE2_2.value in args.chain_type:
            ans = optimize_intersections(QuerDAG.TYPE2_2.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE1_3_joint.value == args.chain_type:
            ans =  get_type13_graph_optimizaton_joint(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE1_3.value == args.chain_type:
            ans = optimize_chains(QuerDAG.TYPE1_3.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE2_3.value == args.chain_type:
            ans = optimize_intersections(QuerDAG.TYPE2_3.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE3_3.value == args.chain_type:
            ans =  get_type33_graph_optimizaton(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)

        if QuerDAG.TYPE4_3.value == args.chain_type:
            ans =  get_type43_graph_optimizaton(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm)
        if 'e' == args.chain_type:
            ans = exhaustive_search_comparison(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, '1_2')
