import torch
import numpy as np
from tqdm import tqdm
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
