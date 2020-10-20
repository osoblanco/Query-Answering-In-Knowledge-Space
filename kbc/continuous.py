import argparse
import pickle
import os.path as osp
from pathlib import Path
import json

from kbc.utils import QuerDAG
from kbc.utils import preload_env
from kbc.metrics import evaluation


def main(args):
    mode = args.dataset_mode

    script_path = osp.dirname(Path(__file__).absolute())

    data_hard_path = osp.join(script_path, 'data', args.dataset,
                              f'{args.dataset}_{mode}_hard.pkl')
    data_complete_path = osp.join(script_path, 'data', args.dataset,
                                  f'{args.dataset}_{mode}_complete.pkl')

    data_hard = pickle.load(open(data_hard_path, 'rb'))
    data_complete = pickle.load(open(data_complete_path, 'rb'))

    # Instantiate singleton KBC object
    preload_env(args.model_path, data_hard, args.chain_type, mode='hard')
    env = preload_env(args.model_path, data_complete, args.chain_type, mode='complete')

    queries = env.keys_hard
    test_ans_hard = env.target_ids_hard
    test_ans = env.target_ids_complete
    chains = env.chains
    kbc = env.kbc

    if args.reg is not None:
        env.kbc.regularizer.weight = args.reg

    disjunctive = args.chain_type in (QuerDAG.TYPE2_2u.value, QuerDAG.TYPE4_3u.value)

    if args.chain_type in (QuerDAG.TYPE1_2.value, QuerDAG.TYPE1_3.value):
        scores = kbc.model.optimize_chains(chains, kbc.regularizer,
                                           max_steps=1000,
                                           t_norm=args.t_norm)

    elif args.chain_type in (QuerDAG.TYPE2_2.value, QuerDAG.TYPE2_2u.value, QuerDAG.TYPE3_3.value):
        scores = kbc.model.optimize_intersections(chains, kbc.regularizer,
                                                  max_steps=1000,
                                                  t_norm=args.t_norm,
                                                  disjunctive=disjunctive)

    elif args.chain_type == QuerDAG.TYPE3_3.value:
        scores = kbc.model.type3_3chain_optimize(chains, kbc.regularizer,
                                                 max_steps=1000,
                                                 t_norm=args.t_norm)

    elif args.chain_type in (QuerDAG.TYPE4_3.value, QuerDAG.TYPE4_3u.value):
        scores = kbc.model.type4_3chain_optimize(chains, kbc.regularizer,
                                                 max_steps=1000,
                                                 t_norm=args.t_norm,
                                                 disjunctive=disjunctive)
    else:
        raise ValueError(f'Uknown query type {args.chain_type}')

    metrics = evaluation(scores, queries, test_ans, test_ans_hard)
    print(metrics)

    model_name = osp.splitext(osp.basename(args.model_path))[0]
    reg_str = f'-{args.reg}' if args.reg is not None else ''
    with open(f'{model_name}-{args.chain_type}{reg_str}-{mode}.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":

    datasets = ['FB15k', 'FB15k-237', 'NELL']
    dataset_modes = ['valid', 'test', 'train']
    chain_types = [t.value for t in QuerDAG]

    t_norms = ['min', 'product']

    parser = argparse.ArgumentParser(description="Query space optimizer namespace")
    parser.add_argument('--model_path', help="The path to the KBC model. Can be both relative and full")
    parser.add_argument('--dataset', choices=datasets, help="Dataset in {}".format(datasets))
    parser.add_argument('--dataset_mode', choices=dataset_modes, default='train',
                        help="Dataset validation mode in {}".format(dataset_modes))
    parser.add_argument('--similarity_metric', default='l2')

    parser.add_argument('--chain_type', choices=chain_types, default=QuerDAG.TYPE1_1.value,
                        help="Chain type experimenting for ".format(chain_types))

    parser.add_argument('--t_norm', choices=t_norms, default='min', help="T-norms available are ".format(t_norms))
    parser.add_argument('--reg', type=float, help='Regularization coefficient', default=None)

    main(parser.parse_args())
