import argparse
import pickle
import json

from kbc.utils import QuerDAG
from kbc.utils import preload_env

from kbc.metrics import evaluation
from kbc.metrics import new_evaluation

def run_all_experiments(kbc_path, dataset_hard, dataset_complete, dataset_name, t_norm='min', candidates=3, scores_normalize=0, timing_threshold=0):
	# for query in ['1_2', '1_3', '2_2', '2_3', '4_3', '3_3', '2_2_disj', '4_3_disj']:
	experiments = ['1_2', '1_3', '2_2', '2_3', '3_3', '4_3', '2_2_disj', '4_3_disj']
	# experiments = ['2_2_disj', '4_3_disj']
	# experiments = ['4_3_disj']
	# experiments = ['3_3', '4_3']
	# experiments = ['2_3']

	print(kbc_path, dataset_name, t_norm, candidates)

	path_entries = kbc_path.split('-')
	rank = path_entries[path_entries.index('rank') + 1] if 'rank' in path_entries else 'None'

	for exp in experiments:
		metrics = query_answer_BF(kbc_path, dataset_hard, dataset_complete, t_norm, exp, candidates, scores_normalize, timing_threshold)

		with open(f'topk_d={dataset_name}_t={t_norm}_e={exp}_rank={rank}_k={candidates}_sn={scores_normalize}.json', 'w') as fp:
			json.dump(metrics, fp)
	return


def query_answer_BF(kbc_path, dataset_hard, dataset_complete, t_norm='min', query_type='1_2', candidates=3, scores_normalize = 0, timing_threshold = 0):

	metrics = {}
	env = preload_env(kbc_path, dataset_hard, query_type, mode = 'hard')
	env = preload_env(kbc_path, dataset_complete, query_type, mode = 'complete')
	env.name = kbc_path.split('/')[-1].split('-')[0]

	if '1' in env.chain_instructions[-1][-1]:
		part1, part2 = env.parts
	elif '2' in env.chain_instructions[-1][-1]:
		part1, part2, part3 = env.parts

	kbc = env.kbc

	scores = kbc.model.query_answering_BF(env, candidates, t_norm=t_norm , batch_size=1, scores_normalize = scores_normalize, timing_threshold = timing_threshold)
	print(scores.shape)

	queries = env.keys_hard
	test_ans_hard = env.target_ids_hard
	test_ans = 	env.target_ids_complete
	# scores = torch.randint(1,1000, (len(queries),kbc.model.sizes[0]),dtype = torch.float).cuda()
	#
	if not timing_threshold:
		metrics = evaluation(scores, queries, test_ans, test_ans_hard)
		print(metrics)

	return metrics


if __name__ == "__main__":

	big_datasets = ['Bio','FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
	datasets = big_datasets
	dataset_modes = ['valid', 'test', 'train']

	chain_types = [QuerDAG.TYPE1_1.value,QuerDAG.TYPE1_2.value,QuerDAG.TYPE2_2.value,QuerDAG.TYPE1_3.value, \
	QuerDAG.TYPE1_3_joint.value, QuerDAG.TYPE2_3.value, QuerDAG.TYPE3_3.value, QuerDAG.TYPE4_3.value,'All','e']

	t_norms = ['min', 'product']
	normalize_choices = ['0', '1']

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
	'--chain_type', choices=chain_types, default=QuerDAG.TYPE1_1.value,
	help="Chain type experimenting for ".format(chain_types)
	)

	parser.add_argument(
	'--scores_normalize', choices=normalize_choices, default='0',
	help="A normalization flag for atomic scores"
	)

	parser.add_argument(
	'--timing_threshold', default='0',
	help="A timing threshhold for query inference"
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

	data_hard_path = args.dataset+'_hard.pkl'
	data_complete_path = args.dataset+'_complete.pkl'

	data_hard = pickle.load(open(data_hard_path,'rb'))
	data_complete = pickle.load(open(data_complete_path,'rb'))

	# query_answer_BF(args.model_path, data_hard, data_complete, args.similarity_metric, args.t_norm, args.chain_type)
	candidates = int(args.candidates)
	run_all_experiments(args.model_path, data_hard, data_complete, args.dataset, t_norm=args.t_norm, candidates=candidates, \
											scores_normalize = int(args.scores_normalize), timing_threshold=int(args.timing_threshold))
