import argparse
import pickle
import os.path as osp

import torch

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.metrics import evaluation


def optimize_chains(query_type, kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min', reg=5e-2):
    try:
        env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
        env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete
        kbc.regularizer.weight = reg

        scores = kbc.model.optimize_chains(chains, kbc.regularizer, max_steps=1000, similarity_metric=similarity_metric, t_norm=t_norm)

        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


def optimize_intersections(query_type, kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min', reg=5e-2):
    try:
        env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
        env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete
        kbc.regularizer.weight = reg

        scores = kbc.model.optimize_intersections(chains, kbc.regularizer, max_steps=1000,
                                                  similarity_metric=similarity_metric, t_norm=t_norm,
                                                  disjunctive=query_type == QuerDAG.TYPE2_2u.value)
        torch.cuda.empty_cache()
        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


def get_type43_graph_optimization(query_type, kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min', reg=5e-2):
    try:
        env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
        env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete
        kbc.regularizer.weight = reg

        scores = kbc.model.type4_3chain_optimize(chains, kbc.regularizer, max_steps=1000, similarity_metric=similarity_metric, t_norm=t_norm,
                                                 disjunctive=query_type == QuerDAG.TYPE4_3u.value)

        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


def get_type33_graph_optimization(kbc_path, dataset_hard, dataset_complete, similarity_metric ='l2', t_norm ='min', reg=5e-2):
    try:
        env = preload_env(kbc_path, dataset_hard, '3_3', mode='hard')
        env = preload_env(kbc_path, dataset_complete, '3_3', mode='complete')

        kbc, chains = env.kbc, env.chains

        queries = env.keys_hard
        test_ans_hard = env.target_ids_hard
        test_ans = env.target_ids_complete
        kbc.regularizer.weight = reg

        scores = kbc.model.type3_3chain_optimize(chains, kbc.regularizer, max_steps=1000, similarity_metric=similarity_metric, t_norm=t_norm)
        torch.cuda.empty_cache()
        print('Evaluating metrics')
        metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
        print(metrics)

    except RuntimeError as e:
        print("Cannot answer the query with a Brute Force: ", e)


if __name__ == "__main__":

    datasets = ['FB15k', 'FB15k-237', 'NELL']
    dataset_modes = ['valid', 'test', 'train']
    similarity_metrics = ['l2', 'Eculidian', 'cosine']

    chain_types = [QuerDAG.TYPE1_1.value,QuerDAG.TYPE1_2.value,QuerDAG.TYPE2_2.value,QuerDAG.TYPE1_3.value,
    QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,'All','e',
                   QuerDAG.TYPE2_2u.value, QuerDAG.TYPE4_3u.value]

    t_norms = ['min', 'product']

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

    parser.add_argument('--reg', type=float, help='Regularization coefficient',
                        default=5e-2)

    args = parser.parse_args()
    mode = args.dataset_mode

    data_hard_path = osp.join('data', args.dataset, f'{args.dataset}_{mode}_hard.pkl')
    data_complete_path = osp.join('data', args.dataset, f'{args.dataset}_{mode}_complete.pkl')

    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))

    if QuerDAG.TYPE1_2.value == args.chain_type:
        ans = optimize_chains(QuerDAG.TYPE1_2.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE2_2.value == args.chain_type:
        ans = optimize_intersections(QuerDAG.TYPE2_2.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE2_2u.value == args.chain_type:
        ans = optimize_intersections(QuerDAG.TYPE2_2u.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE1_3.value == args.chain_type:
        ans = optimize_chains(QuerDAG.TYPE1_3.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE2_3.value == args.chain_type:
        ans = optimize_intersections(QuerDAG.TYPE2_3.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE3_3.value == args.chain_type:
        ans = get_type33_graph_optimization(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)

    if QuerDAG.TYPE4_3.value == args.chain_type:
        ans = get_type43_graph_optimization(QuerDAG.TYPE4_3.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)
    if QuerDAG.TYPE4_3u.value == args.chain_type:
        ans = get_type43_graph_optimization(QuerDAG.TYPE4_3u.value, args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.reg)
