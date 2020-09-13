import enum
from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

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
    TYPE1_3 = "1_3"
    TYPE2_3 = "2_3"
    TYPE3_3 = "3_3"
    TYPE4_3 = "4_3"
    TYPE1_3_joint = '1_3_joint'


class DynKBCSingleton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if DynKBCSingleton.__instance == None:
            DynKBCSingleton()
        return DynKBCSingleton.__instance


    def set_attr(self, kbc, chains, parts, target_ids, keys, chain_instructions, graph_type, lhs_norm ):
        self.kbc = kbc
        self.chains = chains
        self.parts = parts
        self.target_ids = target_ids
        self.keys = keys
        self.lhs_norm = lhs_norm
        self.chain_instructions = chain_instructions
        self.graph_type = graph_type
        self.__instance = self

    def __init__(self,kbc = None, chains = None , parts = None, \
    target_ids = None, keys = None, lhs_norm = None, chain_instructions = None, graph_type = None):
        """ Virtually private constructor. """
        if DynKBCSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DynKBCSingleton.kbc = kbc
            DynKBCSingleton.chains = chains
            DynKBCSingleton.parts = parts
            DynKBCSingleton.target_ids = target_ids
            DynKBCSingleton.keys = keys

            DynKBCSingleton.lhs_norm = lhs_norm
            DynKBCSingleton.graph_type = graph_type
            DynKBCSingleton.chain_instructions = chain_instructions
            DynKBCSingleton.__instance = self
