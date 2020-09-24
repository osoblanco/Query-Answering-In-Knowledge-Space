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
from kbc.utils import create_instructions
from kbc.utils import preload_env


from kbc.metrics import average_percentile_rank
from kbc.metrics import norm_comparison
from kbc.metrics import evaluation





def query_answer_BF(kbc_path, dataset_hard, dataset_complete, similarity_metric = 'l2', t_norm = 'min', query_type = '1_2'):
	try:

		#
		# print(len(dataset_hard.type1_2chain))
		# print(len(dataset_hard.type2_2chain))
		# print(len(dataset_hard.type2_3chain))
		# print(len(dataset_hard.type1_3chain))
		# print(len(dataset_hard.type3_3chain))
		# print(len(dataset_hard.type4_3chain))

		env = preload_env(kbc_path, dataset_hard, query_type, mode = 'hard')
		env = preload_env(kbc_path, dataset_complete, query_type, mode = 'complete')

		print(len(env.target_ids_hard))

		if '1' in env.chain_instructions[-1][-1]:
			part1, part2 = env.parts
		elif '2' in env.chain_instructions[-1][-1]:
			part1, part2, part3 = env.parts

		# part1, part2, part3 = env.parts

		# keys = env.keys
		# target_ids,lhs_norm  = env.target_ids, env.lhs_norm
		kbc, chains = env.kbc, env.chains
		#

		scores =  kbc.model.query_answering_BF(env , kbc.regularizer, 2)
		torch.cuda.empty_cache()
		#
		# hits = hits_at_k(indices_rankedby_distances, target_ids, keys, hits = [1,3])

		# APR = average_percentile_rank(indices_rankedby_distances,target_ids, keys)

		queries = env.keys_hard
		test_ans_hard = env.target_ids_hard
		test_ans = 	env.target_ids_complete
		# scores = torch.randint(1,1000, (len(queries),kbc.model.sizes[0]),dtype = torch.float).cuda()
        #
		metrics = evaluation(scores, queries, test_ans, test_ans_hard, env)
		print(metrics)


	except RuntimeError as e:
		print("Cannot answer the query with a Brute Force: ", e)
		return None
	return None


if __name__ == "__main__":

	big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
	datasets = big_datasets
	dataset_modes = ['valid', 'test', 'train']
	similarity_metrics = ['l2', 'Eculidian', 'cosine']

	chain_types = [QuerDAG.TYPE1_1.value,QuerDAG.TYPE1_2.value,QuerDAG.TYPE2_2.value,QuerDAG.TYPE1_3.value, \
	QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,'All','e']

	t_norms = ['min','product']

	parser = argparse.ArgumentParser(
	description="Query Answering BF namespace"
	)

	parser.add_argument(
	'--model_path',
	help="The path to the KBC model. Can be both relative and full"
	)

	parser.add_argument(
	'--dataset',
	help="The pickled Dataset name containing the chains"
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

	data_hard_path = args.dataset+'_hard.pkl'
	data_complete_path = args.dataset+'_complete.pkl'

	data_hard = pickle.load(open(data_hard_path,'rb'))
	data_complete = pickle.load(open(data_complete_path,'rb'))

	query_answer_BF(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.chain_type)
