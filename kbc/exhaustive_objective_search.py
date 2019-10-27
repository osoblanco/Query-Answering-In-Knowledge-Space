import torch


import numpy as np

from kbc.learn import kbc_model_load
from kbc.learn import dataset_to_query

from kbc.chain_dataset import ChaineDataset
from kbc.chain_dataset import Chain
from kbc.utils import QuerDAG


def exhastive_objective_search(model, chains: List, regularizer: Regularizer, similarity_metric : str = 'l2', t_norm: str = 'min', graph_type : str=QuerDAG.TYPE1_2.value):
    try:

        if 'cp' in self.model_type().lower():
            candidates = self.rhs
        elif 'complex' in self.model_type().lower():
            candidates = self.embeddings[0].weight.data

        if QuerDAG.TYPE1_2.value in graph_type:
            lhs_1,rel_1,rel_2,rhs_2 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE1_2.value)
        if QuerDAG.TYPE2_2.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE2_2.value)
        if QuerDAG.TYPE1_3.value in graph_type:
            lhs_1,rel_1,rhs_1,lhs_2,rel_2,rel_3,rhs_3 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE1_3.value)
        if QuerDAG.TYPE2_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE2_3.value)
        if QuerDAG.TYPE3_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE3_3.value)
        if QuerDAG.TYPE4_3.value in graph_type:
            lhs_1,rel_1,lhs_2,rel_2,rel_3,rhs_3 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE4_3.value)

        with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

            best_candidates = []
            for i in range(candidates.shape[0]):
                best_candidate_loss = 100000
                best_candidate_id = 0
                best_candidate = candidates[best_candidate_id]

                for j in range(candidates.shape[0]):

                    if QuerDAG.TYPE1_2.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))
                        score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))

                        l_reg_2 = regularizer.forward((best_candidate[j:j+1], rel_2[i:i+1], rhs_2[i:i+1]))
                        score_2 = -(self.score_emb(best_candidate[j:j+1], rel_2[i:i+1], rhs_2[i:i+1]))

                        if 'min' in t_norm.lower():
                            loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)
                        elif 'prod' in t_norm.lower():
                            loss = (score_1 +l_reg_1) * (score_2 + l_reg_2)

                    elif QuerDAG.TYPE2_2.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))
                        score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]) )
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        score_2 = -(self.score_emb(lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)

                    elif QuerDAG.TYPE1_3.value in graph_type:
                        # l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], rhs_1[i:i+1][i:i+1]))
                        # score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], rhs_1[i:i+1]))[i:i+1]
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        score_2 = -(self.score_emb(lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]) )
                        l_reg_3 = regularizer.forward((best_candidate[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        score_3 = -(self.score_emb(best_candidate[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        loss = torch.min(torch.stack([score_2,score_3])) - (-l_reg_3 - l_reg_2 )

                    elif QuerDAG.TYPE2_3.value in graph_type:

                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1][i:i+1], best_candidate[j:j+1]))
                        score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        score_2 = -(self.score_emb(lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]) )
                        l_reg_3 = regularizer.forward((lhs_3[i:i+1], rel_3[i:i+1], best_candidate[j:j+1]))
                        score_3 = -(self.score_emb(lhs_3[i:i+1], rel_3[i:i+1], best_candidate[j:j+1]))

                    elif QuerDAG.TYPE3_3.value in graph_type:
                        # l_reg_1 = regularizer.forward((lhs_1, rel_1, lhs_2)) *0
                        # score_1 = -(self.score_emb(lhs_1, rel_1, lhs_2))*0

                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        score_2 = -(self.score_emb(lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]) )
                        l_reg_3 = regularizer.forward((lhs_3[i:i+1], rel_3[i:i+1], best_candidate[j:j+1]))
                        score_3 = -(self.score_emb(lhs_3[i:i+1], rel_3[i:i+1], best_candidate[j:j+1]))
                        loss = torch.min(torch.stack([score_2,score_3])) - ( - l_reg_2 - l_reg_3 )

                    elif QuerDAG.TYPE4_3.value in graph_type:
                        l_reg_1 = regularizer.forward((lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))
                        score_1 = -(self.score_emb(lhs_1[i:i+1], rel_1[i:i+1], best_candidate[j:j+1]))
                        l_reg_2 = regularizer.forward((lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]))
                        score_2 = -(self.score_emb(lhs_2[i:i+1], rel_2[i:i+1], best_candidate[j:j+1]) )
                        l_reg_3 = regularizer.forward((best_candidate[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))
                        score_3 = -(self.score_emb(best_candidate[j:j+1], rel_3[i:i+1], rhs_3[i:i+1]))

                        loss = torch.min(torch.stack([score_1,score_2,score_3])) - (-l_reg_1 - l_reg_2 - l_reg_3 )


                    if best_candidate_loss > loss:
                        best_candidate_loss = loss
                        best_candidate_id = j
                        best_candidate = candidates[best_candidate_id]

                best_candidates.append((best_candidate_id,best_candidate,best_candidate_loss))

                bar.update(1)
                bar.set_postfix(loss=f'{loss.item():.6f}')


    except RuntimeError as e:
        print("Exhastive search Completed with error: ",e)
        return None
    return best_candidates
