# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import torch
from torch import nn
from torch import optim
from kbc.regularizers import Regularizer
import tqdm
import enum

import gc

from kbc.utils import QuerDAG

import matplotlib.pyplot as plt

class KBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def get_queries_separated(self, queries: torch.Tensor):
            pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    @abstractmethod
    def score_emb(self, lhs: torch.Tensor, rel: torch.Tensor, rhs: torch.Tensor):
        pass

    def model_type(self):
        pass


    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    q = self.get_queries(these_queries)

                    scores = q @ rhs
                    targets = self.score(these_queries)

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [int(x - c_begin) for x in filter_out if c_begin <= x < c_begin + chunk_size]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6

                    ranks[b_begin:b_begin + batch_size] += torch.sum((scores >= targets).float(), dim=1).cpu()

                    b_begin += batch_size

                c_begin += chunk_size

        return ranks

    def projected_gradient_descent(self, query: tuple,regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001,
                                    similarity_metric : str = 'l2'):
        try:

            try:
                lhs = query[0].clone().detach().requires_grad_(False).to(query[0].device)
                pred = query[1].clone().detach().requires_grad_(False).to(query[1].device)
            except:
                print("Cuda Memory not enough trying a hack")
                lhs = query[0]
                pred = query[1]


            obj_guess = torch.rand(lhs.shape, requires_grad=True, device=lhs.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess = obj_guess.clone().detach().requires_grad_(True).to(lhs.device)


            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                losses = []

                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()
                    l_reg = regularizer.forward((lhs, pred, obj_guess))
                    loss = -(self.score_emb(lhs, pred, obj_guess) - l_reg)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')
                    losses.append(loss.item())

                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))



                #print(losses)

                torch.cuda.empty_cache()
                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)
                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)
                else:
                    print("Choose model type from cp or complex please")
                    raise


        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess, closest_map, indices_rankedby_distances

    def __get_chains__(self, chains:List , graph_type:str =QuerDAG.TYPE1_2.value):
        try:
            if '2' in graph_type[-1]:
                chain1, chain2 = chains
            elif '3' in graph_type[-1]:
                chain1, chain2, chain3 = chains

            if QuerDAG.TYPE1_2.value in graph_type:
                try:
                    lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                    rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)

                    rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)
                    rhs_2 = chain2[2].clone().detach().requires_grad_(False).to(chain2[2].device)
                except:
                    print("Cuda Memory not enough trying a hack")
                    lhs_1 = chain1[0]
                    rel_1 = chain1[1]

                    lhs_2 = chain2[1]
                    rel_2 = chain2[2]

                raw_chain = [lhs_1,rel_1,rel_2,rhs_2]

            elif QuerDAG.TYPE2_2.value in graph_type:
                try:
                    lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                    rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)

                    lhs_2 = chain2[0].clone().detach().requires_grad_(False).to(chain2[0].device)
                    rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)

                except:
                    print("Cuda Memory not enough trying a hack")
                    lhs_1 = chain1[0]
                    rel_1 = chain1[1]

                    lhs_2 = chain2[0]
                    rel_2 = chain2[1]

                raw_chain = [lhs_1,rel_1,lhs_2,rel_2]

            elif QuerDAG.TYPE1_3.value in graph_type:
                try:
                    lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                    rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)
                    rhs_1 = chain1[2].clone().detach().requires_grad_(False).to(chain1[2].device)

                    lhs_2 = chain2[0].clone().detach().requires_grad_(False).to(chain2[0].device)
                    rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)

                    rel_3 = chain3[1].clone().detach().requires_grad_(False).to(chain3[1].device)
                    rhs_3 = chain3[2].clone().detach().requires_grad_(False).to(chain3[2].device)

                except:
                    print("Cuda Memory not enough trying a hack")
                    lhs_1 = chain1[0]
                    rel_1 = chain1[1]
                    rhs_1 = chain1[2]

                    lhs_2 = chain2[0]
                    rel_2 = chain2[1]

                    rel_3 = chain3[1]
                    rhs_3 = chain3[2]

                raw_chain = [lhs_1,rel_1,rhs_1,lhs_2,rel_2,rel_3,rhs_3]

            elif QuerDAG.TYPE2_3.value in graph_type or QuerDAG.TYPE3_3.value in graph_type:
                try:
                    lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                    rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)

                    lhs_2 = chain2[0].clone().detach().requires_grad_(False).to(chain2[0].device)
                    rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)

                    lhs_3 = chain3[0].clone().detach().requires_grad_(False).to(chain3[0].device)
                    rel_3 = chain3[1].clone().detach().requires_grad_(False).to(chain3[1].device)

                except:
                    print("Cuda Memory not enough trying a hack")
                    lhs_1 = chain1[0]
                    rel_1 = chain1[1]

                    lhs_2 = chain2[0]
                    rel_2 = chain2[1]

                    lhs_3 = chain3[0]
                    rel_3 = chain3[1]

                raw_chain = [lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3]

            elif QuerDAG.TYPE4_3.value in graph_type:
                try:
                    lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                    rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)

                    lhs_2 = chain2[0].clone().detach().requires_grad_(False).to(chain2[0].device)
                    rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)

                    rel_3 = chain3[1].clone().detach().requires_grad_(False).to(chain3[1].device)
                    rhs_3 = chain3[2].clone().detach().requires_grad_(False).to(chain3[2].device)

                except:
                    print("Cuda Memory not enough trying a hack")
                    lhs_1 = chain1[0]
                    rel_1 = chain1[1]

                    lhs_2 = chain2[0]
                    rel_2 = chain2[1]

                    rel_3 = chain3[1]
                    rhs_3 = chain3[2]

                raw_chain = [lhs_1,rel_1,lhs_2,rel_2,rel_3,rhs_3]

        except RuntimeError as e:
            print("Cannot Get chains with Error: ",e)
            return None

        return raw_chain

    def exhastive_objective_search(self, chains: List, regularizer: Regularizer,candidates: int = 1, max_steps: int = 20, \
                                    step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min', graph_type : str=QuerDAG.TYPE1_2.value):
        try:
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


        except RuntimeError as e:
            print("Exhastive search Completed with error: ",e)
            return None
        return None

    def type1_2chain_optimize(self, chains: List, regularizer: Regularizer,candidates: int = 1,
                            max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,rel_2,rhs_2 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE1_2.value)
            obj_guess = torch.rand(rhs_2.shape, requires_grad=True, device=rhs_2.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess = obj_guess.clone().detach().requires_grad_(True).to(rhs_2.device)


            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            losses = []

            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    l_reg_1 = regularizer.forward((lhs_1, rel_1, obj_guess))
                    score_1 = -(self.score_emb(lhs_1, rel_1, obj_guess))


                    l_reg_2 = regularizer.forward((obj_guess, rel_2, rhs_2))
                    score_2 = -(self.score_emb(obj_guess, rel_2, rhs_2))

                    if 'min' in t_norm.lower():
                        loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)
                    elif 'prod' in t_norm.lower():
                        loss = (score_1 +l_reg_1) * (score_2 + l_reg_2)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())

                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))


                torch.cuda.empty_cache()
                lhs_1 = None
                rel_1 = None

                rel_2 = None
                rhs_2 = None

                torch.cuda.empty_cache()
                gc.collect()

                #print(losses)

                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)
                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)
                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess, closest_map, indices_rankedby_distances

    def type2_2chain_optimize(self, chains: List, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,lhs_2,rel_2 = self.__get_chains__(chains, graph_type =QuerDAG.TYPE2_2.value)
            obj_guess = torch.rand(lhs_2.shape, requires_grad=True, device=lhs_2.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess = obj_guess.clone().detach().requires_grad_(True).to(lhs_2.device)


            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            losses = []

            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    l_reg_1 = regularizer.forward((lhs_1, rel_1, obj_guess))
                    score_1 = -(self.score_emb(lhs_1, rel_1, obj_guess) )


                    l_reg_2 = regularizer.forward((lhs_2, rel_2, obj_guess))
                    score_2 = -(self.score_emb(lhs_2, rel_2, obj_guess))

                    loss = torch.min(score_1,score_2) - (-l_reg_1 - l_reg_2)

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())


                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))


                torch.cuda.empty_cache()
                lhs_1 = None
                rel_1 = None

                lhs_2 = None
                rel_2 = None

                torch.cuda.empty_cache()
                gc.collect()

                #print(losses)

                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)
                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)
                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess, closest_map, indices_rankedby_distances


    def type1_3chain_optimize_joint(self, chain1: tuple, chain2: tuple, chain3: tuple, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:
            try:
                lhs_1 = chain1[0].clone().detach().requires_grad_(False).to(chain1[0].device)
                rel_1 = chain1[1].clone().detach().requires_grad_(False).to(chain1[1].device)

                rel_2 = chain2[1].clone().detach().requires_grad_(False).to(chain2[1].device)

                rel_3 = chain3[1].clone().detach().requires_grad_(False).to(chain3[1].device)
                rhs_3 = chain3[2].clone().detach().requires_grad_(False).to(chain3[2].device)

            except:
                print("Cuda Memory not enough trying a hack")
                lhs_1 = chain1[0]
                rel_1 = chain1[1]

                rel_2 = chain2[1]

                rel_3 = chain3[1]
                rhs_3 = chain3[2]

            obj_guess_1 = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess_1= obj_guess_1.clone().detach().requires_grad_(True).to(lhs_1.device)

            obj_guess_2 = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess_2= obj_guess_1.clone().detach().requires_grad_(True).to(lhs_1.device)

            optimizer = optim.Adam([obj_guess_1,obj_guess_2], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)


            losses = []
            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    l_reg_1 = regularizer.forward((lhs_1, rel_1, obj_guess_1))
                    score_1 = -(self.score_emb(lhs_1, rel_1, obj_guess_1))

                    l_reg_2 = regularizer.forward((obj_guess_1, rel_2, obj_guess_2))
                    score_2 = -(self.score_emb(obj_guess_1, rel_2, obj_guess_2) )

                    l_reg_3 = regularizer.forward((obj_guess_2, rel_3, rhs_3))
                    score_3 = -(self.score_emb(obj_guess_2, rel_3, rhs_3))

                    loss = torch.min(torch.stack([score_1,score_2,score_3])) - (-l_reg_3 - l_reg_2 - l_reg_1) \
                    + 0.0001*((torch.dist(obj_guess_1, lhs_1, 2)) + (torch.dist(obj_guess_2, rhs_3, 2)))/2.0

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())


                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))


                torch.cuda.empty_cache()
                gc.collect()

                #print(losses)

                if 'cp' in self.model_type().lower():
                    closest_map_1, indices_rankedby_distances_1 = self.__closest_matrix__(obj_guess_1,self.rhs,similarity_metric)

                    torch.cuda.empty_cache()
                    gc.collect()

                    closest_map_2, indices_rankedby_distances_2 = self.__closest_matrix__(obj_guess_2,self.rhs,similarity_metric)

                elif 'complex' in self.model_type().lower():
                    closest_map_1, indices_rankedby_distances_1 = self.__closest_matrix__(obj_guess_1,self.embeddings[0].weight.data,similarity_metric)
                    torch.cuda.empty_cache()
                    gc.collect()
                    closest_map_2, indices_rankedby_distances_2 = self.__closest_matrix__(obj_guess_2,self.embeddings[0].weight.data,similarity_metric)

                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess_1, obj_guess_2, closest_map_1,closest_map_2, indices_rankedby_distances_1,indices_rankedby_distances_2


    def type1_3chain_optimize(self, chains: List, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,rhs_1,lhs_2,rel_2,rel_3,rhs_3 = self.__get_chains__(chains, graph_type =QuerDAG.TYPE1_3.value)

            obj_guess = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess= obj_guess.clone().detach().requires_grad_(True).to(lhs_1.device)

            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)


            losses = []

            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()
                    # l_reg_1 = regularizer.forward((lhs_1, rel_1, rhs_1))
                    # score_1 = -(self.score_emb(lhs_1, rel_1, rhs_1))

                    l_reg_2 = regularizer.forward((lhs_2, rel_2, obj_guess))
                    score_2 = -(self.score_emb(lhs_2, rel_2, obj_guess) )

                    l_reg_3 = regularizer.forward((obj_guess, rel_3, rhs_3))
                    score_3 = -(self.score_emb(obj_guess, rel_3, rhs_3))

                    loss = torch.min(torch.stack([score_2,score_3])) - (-l_reg_3 - l_reg_2 )

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')
                    losses.append(loss.item())

                if i != max_steps:
                    bar.update(max_steps-i +1)
                    print("\n\n Search converged early after {} iterations".format(i))

                #print(losses)
                torch.cuda.empty_cache()
                gc.collect()

                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)

                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)

                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess,closest_map,indices_rankedby_distances

    def type2_3chain_optimize(self, chains:List, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = self.__get_chains__(chains , graph_type  =QuerDAG.TYPE2_3.value)
            obj_guess = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess= obj_guess.clone().detach().requires_grad_(True).to(lhs_1.device)

            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            losses = []
            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    l_reg_1 = regularizer.forward((lhs_1, rel_1, obj_guess))
                    score_1 = -(self.score_emb(lhs_1, rel_1, obj_guess))

                    l_reg_2 = regularizer.forward((lhs_2, rel_2, obj_guess))
                    score_2 = -(self.score_emb(lhs_2, rel_2, obj_guess) )

                    l_reg_3 = regularizer.forward((lhs_3, rel_3, obj_guess))
                    score_3 = -(self.score_emb(lhs_3, rel_3, obj_guess))

                    loss = torch.min(torch.stack([score_1,score_2,score_3])) - (-l_reg_1 - l_reg_2 - l_reg_3 )

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())


                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))


                torch.cuda.empty_cache()
                gc.collect()

                #print(losses)

                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)

                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)

                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess,closest_map,indices_rankedby_distances


    def type3_3chain_optimize(self, chains: List, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = self.__get_chains__(chains, graph_type =QuerDAG.TYPE3_3.value)

            obj_guess = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess= obj_guess.clone().detach().requires_grad_(True).to(lhs_1.device)

            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            losses = []
            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    # l_reg_1 = regularizer.forward((lhs_1, rel_1, lhs_2)) *0
                    # score_1 = -(self.score_emb(lhs_1, rel_1, lhs_2))*0

                    l_reg_2 = regularizer.forward((lhs_2, rel_2, obj_guess))
                    score_2 = -(self.score_emb(lhs_2, rel_2, obj_guess) )

                    l_reg_3 = regularizer.forward((lhs_3, rel_3, obj_guess))
                    score_3 = -(self.score_emb(lhs_3, rel_3, obj_guess))

                    loss = torch.min(torch.stack([score_2,score_3])) - ( - l_reg_2 - l_reg_3 )

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())


                if i != max_steps:
                    bar.update(max_steps-i +1)


                    print("\n\n Search converged early after {} iterations".format(i))

                torch.cuda.empty_cache()
                gc.collect()
                #print(losses)

                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)

                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)

                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess,closest_map,indices_rankedby_distances

    def type4_3chain_optimize(self, chains: List, regularizer: Regularizer,candidates: int = 1,
                                    max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min' ):
        try:

            lhs_1,rel_1,lhs_2,rel_2,rel_3,rhs_3 = self.__get_chains__(chains , graph_type =QuerDAG.TYPE4_3.value)
            obj_guess = torch.rand(lhs_1.shape, requires_grad=True, device=lhs_1.device)*1e-5 #lhs.clone().detach().requires_grad_(True).to(lhs.device)
            obj_guess= obj_guess.clone().detach().requires_grad_(True).to(lhs_1.device)

            optimizer = optim.Adam([obj_guess], lr=0.1)

            prev_loss =  torch.tensor([1000.], dtype = torch.float)
            loss = torch.tensor([999.],dtype=torch.float)

            losses = []

            with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:

                i =1
                while i <= max_steps and (prev_loss - loss)>1e-30:

                    prev_loss = loss.clone()

                    l_reg_1 = regularizer.forward((lhs_1, rel_1, obj_guess))
                    score_1 = -(self.score_emb(lhs_1, rel_1, obj_guess))

                    l_reg_2 = regularizer.forward((lhs_2, rel_2, obj_guess))
                    score_2 = -(self.score_emb(lhs_2, rel_2, obj_guess) )

                    l_reg_3 = regularizer.forward((obj_guess, rel_3, rhs_3))
                    score_3 = -(self.score_emb(obj_guess, rel_3, rhs_3))

                    loss = torch.min(torch.stack([score_1,score_2,score_3])) - (-l_reg_1 - l_reg_2 - l_reg_3 )

                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    i+=1
                    bar.update(1)
                    bar.set_postfix(loss=f'{loss.item():.6f}')

                    losses.append(loss.item())


                if i != max_steps:
                    bar.update(max_steps-i +1)
                    print("\n\n Search converged early after {} iterations".format(i))


                torch.cuda.empty_cache()
                gc.collect()

                #print(losses)
                if 'cp' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.rhs,similarity_metric)

                elif 'complex' in self.model_type().lower():
                    closest_map, indices_rankedby_distances = self.__closest_matrix__(obj_guess,self.embeddings[0].weight.data,similarity_metric)

                else:
                    print("Choose model type from cp or complex please")
                    raise

        except RuntimeError as e:
            print("Cannot optimize the queries with error {}".format(str(e)))
            return None

        return obj_guess,closest_map,indices_rankedby_distances



    def __expanded_pairwise_distances__(self,x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''

        dist = None
        try:
            x_norm = (x**2).sum(1).view(-1, 1)
            if y is not None:
                y_norm = (y**2).sum(1).view(1, -1)
            else:
                y = x
                y_norm = x_norm.view(1, -1)

            dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        except RuntimeError as e:
            print("Cannot find Pairwise distance with error {}".format(str(e)))
            return None

        return dist

    def __closest_matrix__(self,obj_matrix: torch.Tensor, search_list: torch.Tensor,
                            similarity_metric : str = 'l2', dist_comput_method: str = 'fast'):

        closest_matrix = []

        try:
            with tqdm.tqdm(total=obj_matrix.shape[0], unit='iter', disable=False) as bar:


                if 'euclid' in similarity_metric.lower() or 'l2' in similarity_metric.lower():


                    if 'fast' in dist_comput_method.lower():

                        dists = self.__expanded_pairwise_distances__(obj_matrix,search_list)
                        if dists is None:
                            print("Trying CPU")
                            dists = self.__expanded_pairwise_distances__(obj_matrix.clone().detach().requires_grad_(False).to("cpu"),search_list.to('cpu'))


                        min_inds = torch.argmin(dists,1)

                        indices_rankedby_distances = dists.clone().detach().requires_grad_(False).to("cpu").sort()[1]

                        dist_mins = dists.gather(1, min_inds.view(-1,1)).reshape(-1)
                        closest_matrix = torch.stack([min_inds.float(), dist_mins],1)

                        bar.update(len(obj_matrix))

                    elif 'stable' in dist_comput_method.lower():
                        for obj_vec in obj_matrix:
                            closest_vec = self.__closest_vector__(obj_vec,search_list, similarity_metric)

                            closest_matrix.append(closest_vec)
                            bar.update(1)
                    else:
                        print("The Method for computing the closest vectors is Unknown, please choose from ['stable', 'fast']")
                        raise

                elif 'cosine' in similarity_metric.lower():

                    #implement the ranking task for cosine similarity
                    indices_rankedby_distances = None

                    for i in range(obj_matrix.shape[0]):
                        closest_vec = self.__closest_vector__(obj_matrix[i:i+1],search_list, similarity_metric)
                        closest_matrix.append(closest_vec)
                        bar.update(1)


        except Exception as e:
            print("Cannot Find the closest Matrix with error {}".format(str(e)))
            return None

        return closest_matrix,indices_rankedby_distances

    def __closest_vector__(self,obj_vec: torch.Tensor, search_list: torch.Tensor, similarity_metric : str = 'l2'):

        closest = None
        try:

            if 'euclid' in similarity_metric.lower() or 'l2' in similarity_metric.lower():
                dists = torch.pairwise_distance(obj_vec, search_list, p=2,eps=1e-8)
            elif 'cos' in similarity_metric.lower():
                dists = torch.cosine_similarity(obj_vec,search_list,eps=1e-8)

            min_ind = torch.argmin(dists)

            closest = (min_ind,search_list[min_ind])

        except Exception as e:
            print("Cannot Find the closest Vector with error {}".format(str(e)))
            return None

        return closest





class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()

        self.sizes = sizes
        self.rank = rank

        self.lhs = nn.Embedding(sizes[0], rank, sparse=True)
        self.rel = nn.Embedding(sizes[1], rank, sparse=True)
        self.rhs = nn.Embedding(sizes[2], rank, sparse=True)

        self.lhs.weight.data *= init_size
        self.rel.weight.data *= init_size
        self.rhs.weight.data *= init_size

    def score(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])

        return torch.sum(lhs * rel * rhs, 1, keepdim=True)

    def score_emb(self, lhs, rel, rhs):
        return torch.mean(torch.sum(lhs * rel * rhs, 1, keepdim=True))

    def forward(self, x):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        return (lhs * rel) @ self.rhs.weight.t(), (lhs, rel, rhs)

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.rhs.weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, x: torch.Tensor):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])

        return (lhs,rel)

    def get_full_embeddigns(self, queries: torch.Tensor):
        lhs = self.lhs(queries[:, 0])
        rel = self.rel(queries[:, 1])
        rhs = self.rhs(queries[:, 2])

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        return self.lhs(queries[:, 0]).data * self.rel(queries[:, 1]).data

    def model_type(self):
        return 'CP'



class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()


        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def score(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.sum(
            (lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
            1, keepdim=True
        )

    def score_emb(self, lhs, rel, rhs):

        lhs_dub = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_dub = rel[:, :self.rank], rel[:, self.rank:]
        rhs_dub = rhs[:, :self.rank], rhs[:, self.rank:]

        return torch.mean(torch.sum(
            (lhs_dub[0] * rel_dub[0] - lhs_dub[1] * rel_dub[1]) * rhs_dub[0] +
            (lhs_dub[0] * rel_dub[1] + lhs_dub[1] * rel_dub[0]) * rhs_dub[1],
            1, keepdim=True))


    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
            (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
            (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
        ), (
            torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
            torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
            torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
        )

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[
            chunk_begin:chunk_begin + chunk_size
        ].transpose(0, 1)

    def get_queries_separated(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])

        return (lhs, rel)

    def get_full_embeddigns(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        rhs = self.embeddings[0](queries[:, 2])

        return (lhs,rel,rhs)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        return torch.cat([
            lhs[0] * rel[0] - lhs[1] * rel[1],
            lhs[0] * rel[1] + lhs[1] * rel[0]
        ], 1)

    def model_type(self):
        return "ComplEx"
