import torch
import numpy as np
from tqdm import tqdm
import logging
import gc
import time


def norm_comparison(queries, obj_guess_raw):
	lhs_norm,  guess_norm = None, None
	try:
		if not torch.is_tensor(queries):
			queries = queries[0]
		if len(list(queries.shape))  == 0:
			lhs_norm = queries
		else:
			print(queries.shape)

			lhs_norm = 0.0
			for lhs_emb in queries[0]:
				lhs_norm+=torch.norm(lhs_emb)

			lhs_norm/= len(queries[0])


		guess_norm = 0.0
		for obj_emb in obj_guess_raw:
			guess_norm+=torch.norm(obj_emb)

		guess_norm/= len(obj_guess_raw)
		print("\n")
		print("The average L2 norm of the trained vectors is {}, while optimized vectors have {}".format(lhs_norm,guess_norm))


	except RuntimeError  as e:
		print("Cannor compare L2 norms with error: ",e)
		return lhs_norm,  guess_norm
	return lhs_norm,  guess_norm



def exclude_answers(prediction, filter_list):

	#Find filter indices
	# start = time.time()
	indices_aranged =  torch.nonzero(prediction[..., None] == torch.tensor(filter_list))
	# end1 = start - time.time()
	# print(f"time p1 {end1}")
	mask = torch.zeros(indices_aranged.shape[0], dtype=torch.long)

	# end2 = start - time.time()
	# print(f"time p2 {end2}")

	remove_indices = indices_aranged.gather(1, mask.view(-1,1)).flatten()

	# end3 = start - time.time()
	# print(f"time p3 {end3}")

	full_indices = torch.arange(prediction.shape[0])

	# end4 = start - time.time()
	# print(f"time p4 {end4}")


	remaining_indices = torch.tensor(np.setdiff1d(full_indices, remove_indices))

	# end5 = start - time.time()
	# print(f"time p5 {end5}")

	return prediction[remaining_indices]



def hits_at_k(indices_rankedby_distances, target_ids, keys,  hits = [1]):

	hits_k = {}
	try:

		correct = torch.zeros(len(hits))

		nb_queries = len(indices_rankedby_distances)

		for i in tqdm(range(nb_queries)):

			key = keys[i]
			predictions = indices_rankedby_distances[i]
			targets = torch.tensor(target_ids[key], dtype=torch.long)
			num_targets = targets.shape[0]

			# Find filter indices
			pred_ans_indices, _ = torch.nonzero(predictions[..., None] == targets, as_tuple=True)
			hits_results = torch.zeros_like(correct)
			ranking = pred_ans_indices - num_targets + 1

			for j, k in enumerate(hits):
				hits_results[j] = (ranking < k).float().mean()

			correct += hits_results


		for ind, k in enumerate(hits):
			hits_k[f"Hits@{k}"] = correct[ind]/float(nb_queries)
			print("Hits@{} at {}".format(k,correct[ind]/float(nb_queries)))

	except RuntimeError as e:
		print("Cannot calculate Hits@K with error: ", e)
		return hits_k
	return hits_k


def evaluation(scores, queries, test_ans, test_ans_hard, env):
	try:

		nentity = len(scores[0])
		step = 0
		logs = []

		# count = 0
		# for key,val in test_ans.items():
		# 	print(key,len(val))
		# 	count += 1
        #
		# 	if count > 5:
		# 		break
        #
		# print("____________")
		# count = 0
		# for key,val in test_ans_hard.items():
		# 	print(key,len(val))
		# 	count += 1
        #
		# 	if count > 5:
		# 		break

		with torch.no_grad():
			for query_id, query in enumerate(tqdm(queries)):
				score = scores[query_id]
				score -= (torch.min(score) - 1)
				ans = test_ans[query]
				hard_ans = test_ans_hard[query]
				all_idx = set(range(nentity))

				false_ans = all_idx - set(ans)
				ans_list = list(ans)
				hard_ans_list = list(hard_ans)
				false_ans_list = list(false_ans)
				ans_idxs = np.array(hard_ans_list)
				vals = np.zeros((len(ans_idxs), nentity))

				vals[np.arange(len(ans_idxs)), ans_idxs] = 1
				axis2 = np.tile(false_ans_list, len(ans_idxs))

				# axis2 == [not_ans_1,...not_ans_k, not_ans_1, ....not_ans_k........] Goes for len(hard_ans) times

				axis1 = np.repeat(range(len(ans_idxs)), len(false_ans))

				# len(false_ans) = nentity - len(ans)
				# axis1 = []
				
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

				vals[axis1, axis2] = 1
				b = torch.Tensor(vals).to(device)
				filter_score = b*score
				argsort = torch.argsort(filter_score, dim=1, descending=True)
				ans_tensor = torch.LongTensor(hard_ans_list).to(device)
				
				argsort = torch.transpose(torch.transpose(argsort, 0, 1) - ans_tensor, 0, 1)
				ranking = (argsort == 0).nonzero()
				ranking = ranking[:, 1]
				ranking = ranking + 1

				ans_vec = np.zeros(nentity)
				ans_vec[ans_list] = 1

				# hits1 = torch.sum((ranking <= 1).to(torch.float)).item()
				# hits3 = torch.sum((ranking <= 3).to(torch.float)).item()
				# hits10 = torch.sum((ranking <= 10).to(torch.float)).item()
				# mr = float(torch.sum(ranking).item())
				# mrr = torch.sum(1./ranking.to(torch.float)).item()

				hits1m = torch.mean((ranking <= 1).to(torch.float)).item()
				hits3m = torch.mean((ranking <= 3).to(torch.float)).item()
				hits10m = torch.mean((ranking <= 10).to(torch.float)).item()
				mrm = torch.mean(ranking.to(torch.float)).item()
				mrrm = torch.mean(1./ranking.to(torch.float)).item()
				num_ans = len(hard_ans_list)


				hits1m_newd = hits1m
				hits3m_newd = hits3m
				hits10m_newd = hits10m
				mrm_newd = mrm
				mrrm_newd = mrrm

				logs.append({
					'MRRm_new': mrrm_newd,
					'MRm_new': mrm_newd,
					'HITS@1m_new': hits1m_newd,
					'HITS@3m_new': hits3m_newd,
					'HITS@10m_new': hits10m_newd,
					'num_answer': num_ans
				})



				del argsort, ranking, filter_score, ans_list, hard_ans_list, score, vals
				torch.cuda.empty_cache()
				gc.collect()

				# print(step)
				if step % 100 == 0:
					logging.info('Evaluating the model... (%d/%d)' % (step, 1000))

				step += 1


		metrics = {}
		num_answer = sum([log['num_answer'] for log in logs])
		for metric in logs[0].keys():
			if metric == 'num_answer':
				continue
			if 'm' in metric:
				metrics[metric] = sum([log[metric] for log in logs])/len(logs)
			else:
				metrics[metric] = sum([log[metric] for log in logs])/num_answer




	except RuntimeError as e:
		import traceback
		print(traceback.print_exc())
		print(e)
		return None
	return metrics






def check_answer(prediction, target):
	check = False
	try:

		predicted_tensor = torch.tensor(prediction, dtype=torch.float64)
		target_tensor = torch.tensor(target, dtype=torch.float64)

		check = predicted_tensor.view(1, -1).eq(target_tensor.view(-1, 1)).sum() > 0
		check = check.item()


	except RuntimeError as e :
		print("Cannot check answer validity with error:  ", e)
		return check
	return check


def average_percentile_rank(indices_rankedby_distances,target_ids, keys, threshold = 1000):
	APR = 0.0
	try:
		for i in range(len(indices_rankedby_distances)):

			key = keys[i]
			targets = target_ids[key]

			correct_ans_indices = [(indices_rankedby_distances[i] == one_target).nonzero()[0].squeeze() for one_target in targets]

			correct_ans_index = min(correct_ans_indices)
			# print(correct_ans_index)
			# print(len(indices_rankedby_distances[i]))

			if correct_ans_index > threshold:
				correct_ans_index = threshold

			APR += 1.0 - float(correct_ans_index) / threshold

		APR /= len(indices_rankedby_distances)

		print("Average Percentile Ranks is: ", APR)


	except RuntimeError as e:
		print("Cannot calculate APR with error:  ", e)
		return APR
	return APR
