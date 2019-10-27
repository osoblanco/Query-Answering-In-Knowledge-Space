import torch

import tqdm

import numpy as np

from kbc.learn import kbc_model_load
from kbc.learn import dataset_to_query

from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import Chain
from kbc.utils import QuerDAG
from kbc.utils import DynKBCSingleton


def exhaustive_objective_search( t_norm: str = 'min', graph_type : str=QuerDAG.TYPE1_2.value):
    try:

        env = DynKBCSingleton.getInstance()
        target_ids,lhs_norm  = env.target_ids, env.lhs_norm
        kbc, chains = env.kbc, env.chains
        regularizer = kbc.regularizer

        if 'cp' in kbc.model.model_type().lower():
            candidates = kbc.model.rhs
        elif 'complex' in kbc.model.model_type().lower():
            candidates = kbc.model.embeddings[0].weight.data

        if QuerDAG.TYPE1_2.value in graph_type:
            lhs_1,rel_1,rel_2,rhs_2 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE1_2.value)
        if QuerDAG.TYPE2_2.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE2_2.value)
        if QuerDAG.TYPE1_3.value in graph_type:
            lhs_1,rel_1,rhs_1,lhs_2,rel_2,rel_3,rhs_3 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE1_3.value)
        if QuerDAG.TYPE2_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE2_3.value)
        if QuerDAG.TYPE3_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE3_3.value)
        if QuerDAG.TYPE4_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,rel_3,rhs_3 = kbc.model.__get_chains__(chains , graph_type =QuerDAG.TYPE4_3.value)


        print(candidates.shape)
        print(lhs_1.shape)
        print(rel_1.shape)

        with tqdm.tqdm(total=lhs_1.shape[0]*candidates.shape[0], unit='iter', disable=False) as bar:

            best_candidates = []
            for i in range(lhs_1.shape[0]):
                best_candidate_loss = 100000
                best_candidate_id = 0
                best_candidate = candidates[best_candidate_id]

                for j in range(candidates.shape[0]):

                    if QuerDAG.TYPE1_2.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))
                        score_1 = -(kbc.model.score_emb(lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))

                        l_reg_2 = regularizer.forward((candidates[j:j+1], rel_2[i:i+1], rhs_2[i:i+1]))
                        score_2 = -(kbc.model.score_emb(candidates[j:j+1], rel_2[i:i+1], rhs_2[i:i+1]))

                        if 'min' in t_norm.lower():
                            loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)
                        elif 'prod' in t_norm.lower():
                            loss = (score_1 +l_reg_1) * (score_2 + l_reg_2)

                    elif QuerDAG.TYPE2_2.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))
                        score_1 = -(kbc.model.score_emb(lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]) )
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        score_2 = -(kbc.model.score_emb(lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)

                    elif QuerDAG.TYPE1_3.value in graph_type:
                        # l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], rhs_1[i:i+1][i:i+1]))
                        # score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], rhs_1[i:i+1]))[i:i+1]
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        score_2 = -(kbc.model.score_emb(lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]) )
                        l_reg_3 = regularizer.forward((candidates[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        score_3 = -(kbc.model.score_emb(candidates[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        loss = torch.min(torch.stack([score_2,score_3])) - (-l_reg_3 - l_reg_2 )

                    elif QuerDAG.TYPE2_3.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1][i:i+1], candidates[j:j+1]))
                        score_1 = -(kbc.model.score_emb(lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        score_2 = -(kbc.model.score_emb(lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]) )
                        l_reg_3 = regularizer.forward((lhs_3[i:i+1], rel_3[i:i+1], candidates[j:j+1]))
                        score_3 = -(kbc.model.score_emb(lhs_3[i:i+1], rel_3[i:i+1], candidates[j:j+1]))

                    elif QuerDAG.TYPE3_3.value in graph_type:
                        # l_reg_1 = regularizer.forward((lhs_1, rel_1, lhs_2)) *0
                        # score_1 = -(self.score_emb(lhs_1, rel_1, lhs_2))*0

                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        score_2 = -(kbc.model.score_emb(lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]) )
                        l_reg_3 = regularizer.forward((lhs_3[i:i+1], rel_3[i:i+1], candidates[j:j+1]))
                        score_3 = -(kbc.model.score_emb(lhs_3[i:i+1], rel_3[i:i+1], candidates[j:j+1]))
                        loss = torch.min(torch.stack([score_2,score_3])) - ( - l_reg_2 - l_reg_3 )

                    elif QuerDAG.TYPE4_3.value in graph_type:
                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))
                        score_1 = -(kbc.model.score_emb(lhs_1[i:i+1], rel_1[i:i+1], candidates[j:j+1]))
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]))
                        score_2 = -(kbc.model.score_emb(lhs_2[i:i+1], rel_2[i:i+1], candidates[j:j+1]) )
                        l_reg_3 = regularizer.forward((candidates[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        score_3 = -(kbc.model.score_emb(candidates[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))

                        loss = torch.min(torch.stack([score_1,score_2,score_3])) - (-l_reg_1 - l_reg_2 - l_reg_3 )


                    if best_candidate_loss > loss:
                        best_candidate_loss = loss
                        best_candidate_id = j
                        best_candidate = candidates[best_candidate_id]

                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                best_candidates.append((best_candidate_id,best_candidate,best_candidate_loss))


    except RuntimeError as e:
        print("Exhastive search Completed with error: ",e)
        return None
    return best_candidates
