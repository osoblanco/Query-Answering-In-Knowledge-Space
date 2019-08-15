import pickle
import os


def bioq_to_kbcq(queries_raw, data_mode, train_threshold = 5e5, test_threshold = 1e4, val_threshold = 3e3):
    queries = None
    try:
        queries = [x[0][1] for x in queries_raw]
        queries = [(x[0], '_'.join(x[1]),x[2]) for x in queries]

        relation_names = [x[1] for x in queries]
        relation_names = list(set(relation_names))
        query_name_to_id_map = {}

        for i in range(len(relation_names)):
            query_name_to_id_map[relation_names[i]] = i

        queries = [[x[0], query_name_to_id_map[x[1]],x[2]] for x in queries]

        if len(queries) > train_threshold and "train" in data_mode.lower():
            print("Unique relations in original train set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in original train set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            queries = queries[:5*(10**5)]

            print("Unique relations in downsampled train set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in downsampled train set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            print("\n")

        if len(queries) > test_threshold and 'test' in data_mode.lower():
            print("Unique relations in original test set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in original test set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            queries = queries[:(10**4)]

            print("Unique relations in downsampled test set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in downsampled test set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            print("\n")

        if len(queries) > val_threshold and 'valid' in data_mode.lower():

            print("Unique relations in original val set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in original val set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            queries = queries[:3*(10**3)]

            print("Unique relations in downsampled val set", len(list(set([x[1] for x in queries]))))
            print("Unique entities in downsampled val set", len(list(set([x[0] for x in queries] + [x[2] for x in queries]))))

            print("\n")


    except RuntimeError as e:
        print(e)
        return None

    return queries

def write_queries(data_mode, queries):
    try:
        data_path = os.path.join(os.getcwd(),"kbc/src_data/Bio")
        lines = ["\t".join([str(i) for i in x]) for x in queries]

        file = open(data_path + "/"+data_mode, "w")
        for line in lines:
            file.write(line)
            file.write("\n")
        file.close()

    except RuntimeError as e:
        print(e)

if __name__ == "__main__":

    try:
        train = pickle.load(open('kbc/src_data/Bio/train_edges.pkl','rb'))
        test = pickle.load(open('kbc/src_data/Bio/test_edges.pkl','rb'))
        val = pickle.load(open('kbc/src_data/Bio/val_edges.pkl','rb'))
    except Exception as e:
        print("Please put the BioData File under kbc/data/Bio ")

    train_q = bioq_to_kbcq(train, 'train')
    test_q = bioq_to_kbcq(test, 'test')
    val_q = bioq_to_kbcq(val, 'valid')

    write_queries("train",train_q)
    write_queries("test",test_q)
    write_queries("valid", val_q)
