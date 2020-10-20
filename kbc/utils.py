import os
import sys
import json
import time
import enum
import logging
import subprocess

import matplotlib.pyplot as plt

from collections import OrderedDict
import xml.etree.ElementTree
import numpy as np
import torch


def create_instructions(chains):
    instructions = []
    try:

        prev_start = None
        prev_end = None

        path_stack = []
        start_flag = True
        for chain_ind, chain in enumerate(chains):

            if start_flag:
                prev_end = chain[-1]
                start_flag = False
                continue


            if prev_end == chain[0]:
                instructions.append(f"hop_{chain_ind-1}_{chain_ind}")
                prev_end = chain[-1]
                prev_start = chain[0]

            elif prev_end == chain[-1]:

                prev_start = chain[0]
                prev_end = chain[-1]

                instructions.append(f"intersect_{chain_ind-1}_{chain_ind}")
            else:
                path_stack.append(([prev_start, prev_end],chain_ind-1))
                prev_start = chain[0]
                prev_end = chain[-1]
                start_flag = False
                continue

            if len(path_stack) > 0:

                path_prev_start = path_stack[-1][0][0]
                path_prev_end = path_stack[-1][0][-1]

                if path_prev_end == chain[-1]:

                    prev_start = chain[0]
                    prev_end = chain[-1]

                    instructions.append(f"intersect_{path_stack[-1][1]}_{chain_ind}")
                    path_stack.pop()
                    continue


    except RuntimeError as e:
        print(e)
        return instructions
    return instructions


def extract(elem, tag, drop_s):
  text = elem.find(tag).text
  if drop_s not in text: raise Exception(text)
  text = text.replace(drop_s, "")
  try:
    return int(text)
  except ValueError:
    return float(text)


def check_gpu():
    d = OrderedDict()
    d["time"] = time.time()

    cmd = ['nvidia-smi', '-q', '-x']
    cmd_out = subprocess.check_output(cmd)
    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")

    util = gpu.find("utilization")
    d["gpu_util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / 11171

    if d["gpu_util"] < 15 and d["mem_used"] < 2816 :
    	msg = 'GPU status: Idle \n'
    else:
    	msg = 'GPU status: Busy \n'

    now = time.strftime("%c")
    return ('\n\nUpdated at %s\n\nGPU utilization: %s %%\nVRAM used: %s %%\n\n%s\n\n' % (now, d["gpu_util"],d["mem_used_per"], msg))


class QuerDAG(enum.Enum):
    TYPE1_1 = "1_1"
    TYPE1_2 = "1_2"
    TYPE2_2 = "2_2"
    TYPE2_2u = "2_2u"
    TYPE1_3 = "1_3"
    TYPE2_3 = "2_3"
    TYPE3_3 = "3_3"
    TYPE4_3 = "4_3"
    TYPE4_3u = "4_3u"
    TYPE1_3_joint = '1_3_joint'


class DynKBCSingleton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DynKBCSingleton.__instance == None:
            DynKBCSingleton()
        return DynKBCSingleton.__instance


    def set_attr(self, kbc, chains, parts, target_ids_hard, keys_hard, target_ids_complete, keys_complete,\
    chain_instructions, graph_type, lhs_norm, cuda ):
        self.kbc = kbc
        self.chains = chains
        self.parts = parts


        self.target_ids_hard = target_ids_hard
        self.keys_hard = keys_hard


        self.target_ids_complete = target_ids_complete
        self.keys_complete = keys_complete

        self.cuda = True
        self.lhs_norm = lhs_norm
        self.chain_instructions = chain_instructions
        self.graph_type = graph_type
        self.__instance = self

    def __init__(self,kbc = None, chains = None , parts = None, \
    target_ids_hard = None, keys_hard = None, target_ids_complete = None, keys_complete = None, \
    lhs_norm = None, chain_instructions = None, graph_type = None, cuda = None):
        """ Virtually private constructor. """
        if DynKBCSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DynKBCSingleton.kbc = kbc
            DynKBCSingleton.chains = chains
            DynKBCSingleton.parts = parts

            DynKBCSingleton.target_ids_hard = target_ids_hard
            DynKBCSingleton.keys_hard = keys_hard

            DynKBCSingleton.target_ids_complete = target_ids_complete
            DynKBCSingleton.keys_complete = keys_complete

            DynKBCSingleton.cuda = True
            DynKBCSingleton.lhs_norm = lhs_norm
            DynKBCSingleton.graph_type = graph_type
            DynKBCSingleton.chain_instructions = chain_instructions
            DynKBCSingleton.__instance = self

    def set_eval_complete(self,target_ids_complete, keys_complete):
            self.target_ids_complete = target_ids_complete
            self.keys_complete = keys_complete
            self.__instance = self


def get_keys_and_targets(parts, targets, graph_type):
    if len(parts) == 2:
        part1, part2 = parts
        part3 = None
    elif len(parts) == 3:
        part1, part2, part3 = parts

    target_ids = {}
    keys = []

    for chain_iter in range(len(part2)):

        if part3:
            key = part1[chain_iter] + part2[chain_iter] + part3[chain_iter]
        else:
            key = part1[chain_iter] + part2[chain_iter]

        key = '_'.join(str(e) for e in key)

        if key not in target_ids:
            target_ids[key] = []
            keys.append(key)

        target_ids[key] = targets[chain_iter]

    return target_ids, keys


def preload_env(kbc_path, dataset, graph_type, mode = "hard"):

    from kbc.learn import kbc_model_load

    env = DynKBCSingleton.getInstance()

    chain_instructions = []
    try:

        kbc, epoch, loss = kbc_model_load(kbc_path)

        for parameter in kbc.model.parameters():
            parameter.requires_grad = False

        keys = []
        target_ids = {}

        if QuerDAG.TYPE1_2.value in graph_type:

            raw = dataset.type1_2chain

            type1_2chain = []
            for i in range(len(raw)):
                type1_2chain.append(raw[i].data)

            part1 = [x['raw_chain'][0] for x in type1_2chain]
            part2 = [x['raw_chain'][1] for x in type1_2chain]

            flattened_part1 =[]
            flattened_part2 = []

            # [[A,b,C][C,d,[Es]]

            targets = []
            for chain_iter in range(len(part2)):
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append(part1[chain_iter])
                targets.append(part2[chain_iter][2])

            part1 = flattened_part1
            part2 = flattened_part2

            target_ids, keys = get_keys_and_targets([part1, part2], targets, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)

            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])

            chains = [chain1,chain2]
            parts = [part1, part2]

        elif QuerDAG.TYPE2_2.value in graph_type:
            if graph_type == QuerDAG.TYPE2_2u.value:
                raw = dataset.type2_2chain_u
            else:
                raw = dataset.type2_2chain

            type2_2chain = []
            for i in range(len(raw)):
                type2_2chain.append(raw[i].data)

            part1 = [x['raw_chain'][0] for x in type2_2chain]
            part2 = [x['raw_chain'][1] for x in type2_2chain]

            flattened_part1 =[]
            flattened_part2 = []

            targets = []
            for chain_iter in range(len(part2)):
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                targets.append(part2[chain_iter][2])


            part1 = flattened_part1
            part2 = flattened_part2

            target_ids, keys = get_keys_and_targets([part1, part2], targets, graph_type)


            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)

            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])
            chains = [chain1,chain2]
            parts = [part1,part2]


        elif QuerDAG.TYPE1_3.value in graph_type:
            raw = dataset.type1_3chain

            type1_3chain = []
            for i in range(len(raw)):
                type1_3chain.append(raw[i].data)


            part1 = [x['raw_chain'][0] for x in type1_3chain]
            part2 = [x['raw_chain'][1] for x in type1_3chain]
            part3 = [x['raw_chain'][2] for x in type1_3chain]


            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            # [A,b,C][C,d,[Es]]
            targets = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                targets.append(part3[chain_iter][2])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3

            target_ids, keys = get_keys_and_targets([part1, part2, part3], targets, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)


            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3]
            parts = [part1, part2, part3]

        elif QuerDAG.TYPE2_3.value in graph_type:
            raw = dataset.type2_3chain

            type2_3chain = []
            for i in range(len(raw)):
                type2_3chain.append(raw[i].data)

            part1 = [x['raw_chain'][0] for x in type2_3chain]
            part2 = [x['raw_chain'][1] for x in type2_3chain]
            part3 = [x['raw_chain'][2] for x in type2_3chain]

            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            targets = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],-(chain_iter+1234)])
                targets.append(part3[chain_iter][2])

            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3

            target_ids, keys = get_keys_and_targets([part1, part2, part3], targets, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)


            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3]
            parts = [part1,part2,part3]

        elif QuerDAG.TYPE3_3.value in graph_type:

            raw = dataset.type3_3chain

            type3_3chain = []
            for i in range(len(raw)):
                type3_3chain.append(raw[i].data)


            part1 = [x['raw_chain'][0] for x in type3_3chain]
            part2 = [x['raw_chain'][1] for x in type3_3chain]
            part3 = [x['raw_chain'][2] for x in type3_3chain]


            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            targets = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],-(chain_iter+1234)])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                targets.append(part3[chain_iter][2])


            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3

            target_ids, keys = get_keys_and_targets([part1, part2, part3], targets, graph_type)


            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)


            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])

            chains = [chain1,chain2,chain3]
            parts = [part1, part2, part3]

        elif QuerDAG.TYPE4_3.value in graph_type:
            if graph_type == QuerDAG.TYPE4_3u.value:
                raw = dataset.type4_3chain_u
            else:
                raw = dataset.type4_3chain

            type4_3chain = []
            for i in range(len(raw)):
                type4_3chain.append(raw[i].data)


            part1 = [x['raw_chain'][0] for x in type4_3chain]
            part2 = [x['raw_chain'][1] for x in type4_3chain]
            part3 = [x['raw_chain'][2] for x in type4_3chain]


            flattened_part1 =[]
            flattened_part2 = []
            flattened_part3 = []

            # [A,r_1,B][C,r_2,B][B, r_3, [D's]]
            targets = []
            for chain_iter in range(len(part3)):
                flattened_part3.append([part3[chain_iter][0],part3[chain_iter][1],-(chain_iter+1234)])
                flattened_part2.append([part2[chain_iter][0],part2[chain_iter][1],part2[chain_iter][2]])
                flattened_part1.append([part1[chain_iter][0],part1[chain_iter][1],part1[chain_iter][2]])
                targets.append(part3[chain_iter][2])


            part1 = flattened_part1
            part2 = flattened_part2
            part3 = flattened_part3

            target_ids, keys = get_keys_and_targets([part1, part2, part3], targets, graph_type)

            if not chain_instructions:
                chain_instructions = create_instructions([part1[0], part2[0], part3[0]])

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            part1 = np.array(part1)
            part1 = torch.tensor(part1.astype('int64'), device=device)

            part2 = np.array(part2)
            part2 = torch.tensor(part2.astype('int64'), device=device)

            part3 = np.array(part3)
            part3 = torch.tensor(part3.astype('int64'), device=device)

            chain1 = kbc.model.get_full_embeddigns(part1)
            chain2 = kbc.model.get_full_embeddigns(part2)
            chain3 = kbc.model.get_full_embeddigns(part3)


            lhs_norm = 0.0
            for lhs_emb in chain1[0]:
                lhs_norm+=torch.norm(lhs_emb)

            lhs_norm/= len(chain1[0])
            chains = [chain1,chain2,chain3]
            parts = [part1,part2,part3]

        else:
            chains = dataset['chains']
            parts = dataset['parts']
            target_ids = dataset['target_ids']
            chain_instructions = create_instructions([parts[0][0], parts[1][0], parts[2][0]])

        if mode == 'hard':
            env.set_attr(kbc, chains, parts, target_ids, keys, None, None, chain_instructions, graph_type, lhs_norm, False )

            # env.set_attr(kbc,chains,parts,target_ids, keys, chain_instructions , graph_type, lhs_norm)
            # def set_attr(kbc, chains, parts, target_ids_hard, keys_hard, target_ids_complete, keys_complete, chain_instructions, graph_type, lhs_norm, cuda ):
        else:
            env.set_eval_complete(target_ids,keys)

    except RuntimeError as e:
        print("Cannot preload environment with error: ", e)
        return env

    return env


def plot_regularization_results():
    reg_values = []
    hits = []
    query_type = '4_3u'
    for f in os.listdir():
        if f.startswith('FB15k-237-model-rank-500-epoch-100-1602506111'):
            values = f.split('-')
            if values[8] == query_type:
                reg = float(values[9])
                rank = values[4]
                reg_values.append(reg)
                results = json.load(open(f))
                hits.append(results['HITS@3m_new'])

    reg_values, hits = zip(*sorted(zip(reg_values, hits)))
    plt.plot(reg_values, hits, label=f'Rank={rank}')
    plt.xscale('log')
    plt.xlabel('Regularization coefficient')
    plt.ylabel('H@3')
    plt.title(f'Results on {query_type} queries')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_regularization_results()
