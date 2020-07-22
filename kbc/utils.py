import enum
from collections import OrderedDict
import json
import subprocess
import sys
import time
import xml.etree.ElementTree

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


    def set_attr(self, kbc, chains, parts, target_ids, lhs_norm):
        self.kbc = kbc
        self.chains = chains
        self.parts = parts
        self.target_ids = target_ids
        self.lhs_norm = lhs_norm
        self.__instance = self

    def __init__(self,kbc = None, chains = None , parts = None, \
    target_ids = None, lhs_norm = None):
        """ Virtually private constructor. """
        if DynKBCSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DynKBCSingleton.kbc = kbc
            DynKBCSingleton.chains = chains
            DynKBCSingleton.parts = parts
            DynKBCSingleton.target_ids = target_ids
            DynKBCSingleton.lhs_norm = lhs_norm
            DynKBCSingleton.__instance = self


#_____________________________________________
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
import random
