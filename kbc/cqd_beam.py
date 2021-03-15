from tqdm import tqdm
import os.path as osp
import argparse
import pickle
import torch
import json

from kbc.utils import QuerDAG
from kbc.utils import preload_env

from kbc.metrics import evaluation

def run_all_experiments(kbc_path, dataset_hard, dataset_complete, dataset_name, t_norm='min', candidates=3, scores_normalize=0):
	experiments = [t.value for t in QuerDAG]

	print(kbc_path, dataset_name, t_norm, candidates)
	path_entries = kbc_path.split('-')
	rank = path_entries[path_entries.index('rank') + 1] if 'rank' in path_entries else 'None'

	for exp in experiments:

		metrics = query_answer_BF(kbc_path, dataset_hard, dataset_complete, t_norm, exp, candidates, scores_normalize)

		with open(f'topk_d={dataset_name}_t={t_norm}_e={exp}_rank={rank}_k={candidates}_sn={scores_normalize}.json', 'w') as fp:
			json.dump(metrics, fp)

	return

def query_link_predictor(env):
	chains = env.chains
	s_emb = chains[0][0]
	p_emb = chains[0][1]

	scores_lst = []
	nb_queries = s_emb.shape[0]
	for i in tqdm(range(nb_queries)):
		with torch.no_grad():
		    batch_s_emb = s_emb[i, :].view(1, -1)
		    batch_p_emb = p_emb[i, :].view(1, -1)
		    batch_chains = [(batch_s_emb, batch_p_emb, None)]
		    batch_scores = env.kbc.model.link_prediction(batch_chains)
		    scores_lst += [batch_scores]

	scores = torch.cat(scores_lst, 0)

	return scores

def query_answer_BF(kbc_path, dataset_hard, dataset_complete, t_norm='min', query_type=QuerDAG.TYPE1_2, candidates=3, scores_normalize = 0):
	env = preload_env(kbc_path, dataset_hard, query_type, mode='hard')
	env = preload_env(kbc_path, dataset_complete, query_type, mode='complete')

	kbc = env.kbc

	if query_type == QuerDAG.TYPE1_1.value:
		scores = query_link_predictor(env)
	else:
		scores = kbc.model.query_answering_BF(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize)

	queries = env.keys_hard
	test_ans_hard = env.target_ids_hard
	test_ans = 	env.target_ids_complete

	metrics = evaluation(scores, queries, test_ans, test_ans_hard)
	print(metrics)

	return metrics


if __name__ == "__main__":

	big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
	datasets = big_datasets
	dataset_modes = ['valid', 'test', 'train']

	chain_types = [QuerDAG.TYPE1_1.value, QuerDAG.TYPE1_2.value, QuerDAG.TYPE2_2.value, QuerDAG.TYPE1_3.value,
				   QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,
				   'All', 'e']

	t_norms = ['min', 'product']
	normalize_choices = ['0', '1']

	parser = argparse.ArgumentParser(
	description="Complex Query Decomposition - Beam"
	)

	parser.add_argument('path', help='Path to directory containing queries')

	parser.add_argument(
	'--model_path',
	help="The path to the KBC model. Can be both relative and full"
	)

	parser.add_argument(
	'--dataset',
	help="The pickled Dataset name containing the chains"
	)

	parser.add_argument(
	'--mode', choices=dataset_modes, default='test',
	help="Dataset validation mode in {}".format(dataset_modes)
	)

	parser.add_argument(
	'--scores_normalize', choices=normalize_choices, default='0',
	help="A normalization flag for atomic scores".format(chain_types)
	)

	parser.add_argument(
	'--t_norm', choices=t_norms, default='min',
	help="T-norms available are ".format(t_norms)
	)

	parser.add_argument(
	'--candidates', default=5,
	help="Candidate amount for beam search"
	)

	args = parser.parse_args()

	dataset = osp.basename(args.path)
	mode = args.mode

	data_hard_path = osp.join(args.path, f'{dataset}_{mode}_hard.pkl')
	data_complete_path = osp.join(args.path, f'{dataset}_{mode}_complete.pkl')

	data_hard = pickle.load(open(data_hard_path, 'rb'))
	data_complete = pickle.load(open(data_complete_path, 'rb'))

	candidates = int(args.candidates)
	run_all_experiments(args.model_path, data_hard, data_complete,
						dataset, t_norm=args.t_norm, candidates=candidates,
						scores_normalize=int(args.scores_normalize))
