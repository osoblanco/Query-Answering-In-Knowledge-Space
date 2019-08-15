import argparse
import torch

from kbc.learn import kbc_model_load
from kbc.learn import dataset_to_query

def get_optimization(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess, closest_map = None, None
    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)
        queries,target_ids = dataset_to_query(kbc.model,dataset,dataset_mode)

        lhs_norm = 0.0
        for lhs_emb in queries[0]:
            lhs_norm+=torch.norm(lhs_emb)

        lhs_norm/= len(queries[0])

        obj_guess_raw, closest_map = kbc.model.projected_gradient_descent(queries, kbc.regularizer,max_steps=1000,similarity_metric=similarity_metric)

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

        # print("\n\nTEST : ", results)

    except RuntimeError as e:
        print("Cannot Optimise the Query space with error: {}".format(str(e)))
        return None, None

    return obj_guess, closest_map



def get_type12_graph_optimizaton(kbc_path, dataset, dataset_mode, similarity_metric = 'l2'):
    obj_guess, closest_map = None, None

    try:

        kbc,epoch,loss = kbc_model_load(kbc_path)
        #Todo (Add newer Version)

    except Exception as e:
        print(e)
        return None

if __name__ == "__main__":

    big_datasets = ['FB15K', 'WN', 'WN18RR', 'FB237', 'YAGO3-10']
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

    obj_guess, closest_map =  get_optimization(args.model_path, args.dataset, args.dataset_mode, args.similarity_metric)
