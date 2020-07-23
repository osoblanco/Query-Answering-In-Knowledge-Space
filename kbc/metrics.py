import torch


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


def hits_at_k(indices_rankedby_distances, target_ids, keys,  hits = [1]):

    hits_k = {}
    try:
        for k in hits:
            predicted_ids = [x[:k] for x in indices_rankedby_distances]
            correct = 0.0
            for i in range(len(predicted_ids)):

                key = keys[i]

                if check_answer(predicted_ids[i], target_ids[key]):
                    correct += 1.0


            hits_k[f"Hits@{k}"] = correct/(len(predicted_ids))
            print("Hits@{} at {}".format(k,correct/(len(predicted_ids))))

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
