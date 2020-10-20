# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional
import math

import torch
from torch import nn
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from kbc.regularizers import Regularizer
import tqdm

import traceback

import gc

from kbc.utils import QuerDAG
from kbc.utils import check_gpu
from kbc.utils import DynKBCSingleton
from kbc.utils import make_batches
from kbc.utils import debug_memory
from kbc.utils import Device

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

	@abstractmethod
	def candidates_score(self, rel: Tensor, arg1: Optional[Tensor], arg2: Optional[Tensor], *args, **kwargs) -> Tuple[ Optional[Tensor ], Optional[Tensor]]:
		pass

	@abstractmethod
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

				#torch.cuda.empty_cache()
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

	def __get_chains__(self, chains: List, graph_type: str = QuerDAG.TYPE1_2.value):
		try:
			if '2' in graph_type[-1]:
				chain1, chain2 = chains
			elif '3' in graph_type[-1]:
				chain1, chain2, chain3 = chains

			if QuerDAG.TYPE1_2.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				rel_2 = chain2[1]

				raw_chain = [lhs_1, rel_1, rel_2]

			elif QuerDAG.TYPE2_2.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				lhs_2 = chain2[0]
				rel_2 = chain2[1]

				raw_chain = [lhs_1, rel_1, lhs_2, rel_2]

			elif QuerDAG.TYPE1_3.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				rel_2 = chain2[1]

				rhs_3 = chain3[1]

				raw_chain = [lhs_1, rel_1, rel_2, rhs_3]

			elif QuerDAG.TYPE2_3.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				lhs_2 = chain2[0]
				rel_2 = chain2[1]

				lhs_3 = chain3[0]
				rel_3 = chain3[1]

				raw_chain = [lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3]

			elif QuerDAG.TYPE3_3.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				rel_2 = chain2[1]

				lhs_2 = chain3[0]
				rel_3 = chain3[1]

				raw_chain = [lhs_1, rel_1, rel_2, lhs_2, rel_3]

			elif QuerDAG.TYPE4_3.value in graph_type:
				lhs_1 = chain1[0]
				rel_1 = chain1[1]

				lhs_2 = chain2[0]
				rel_2 = chain2[1]

				rel_3 = chain3[1]

				raw_chain = [lhs_1, rel_1, lhs_2, rel_2, rel_3]

		except RuntimeError as e:
			print("Cannot Get chains with Error: ", e)
			return None

		return raw_chain


	def optimize_chains(self, chains: List, regularizer: Regularizer, candidates: int = 1,
						max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min'):
		try:
			if len(chains) == 2:
				lhs_1, rel_1, rel_2 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_2.value)
			elif len(chains) == 3:
				lhs_1, rel_1, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE1_3.value)
			else:
				raise ValueError(f'Invalid number of chains: {len(chains)}')

			obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			params = [obj_guess_1, obj_guess_2]
			if len(chains) == 3:
				obj_guess_3 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
				params.append(obj_guess_3)
			optimizer = optim.Adam(params, lr=0.1)

			prev_loss_value = 1000
			loss_value = 999
			losses = []

			with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
				i = 0
				while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
					prev_loss_value = loss_value

					score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
					score_2, factors_2 = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
					factors = [factors_1[2], factors_2[2]]

					atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

					if len(chains) == 3:
						score_3, factors_3 = self.score_emb(obj_guess_2, rel_3, obj_guess_3)
						factors.append(factors_3[2])
						atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

					guess_regularizer = regularizer(factors)

					t_norm = torch.prod(atoms, dim=1)
					loss = -t_norm.mean() + guess_regularizer

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					i += 1
					bar.update(1)
					bar.set_postfix(loss=f'{loss.item():.6f}')

					loss_value = loss.item()
					losses.append(loss_value)

				if i != max_steps:
					bar.update(max_steps - i + 1)
					bar.close()
					print(
						"Search converged early after {} iterations".format(i))

				if len(chains) == 2:
					guess = obj_guess_2
				else:
					guess = obj_guess_3

				with torch.no_grad():
					if len(chains) == 2:
						score_2 = self.forward_emb(obj_guess_1, rel_2)
						atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2), dim=-1))
					else:
						score_3 = self.forward_emb(obj_guess_2, rel_3)
						atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3),dim=-1))

					t_norm = torch.prod(atoms, dim=-1)

					scores = t_norm

		except RuntimeError as e:
			print("Cannot optimize the queries with error {}".format(str(e)))
			return None

		return scores

	def optimize_intersections(self, chains: List, regularizer: Regularizer, candidates: int = 1,
							   max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min',
							   disjunctive=False):
		try:
			if len(chains) == 2:
				raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_2.value)
				lhs_1, rel_1, lhs_2, rel_2 = raw_chain
			elif len(chains) == 3:
				raw_chain = self.__get_chains__(chains, graph_type=QuerDAG.TYPE2_3.value)
				lhs_1, rel_1, lhs_2, rel_2, lhs_3, rel_3 = raw_chain
			else:
				raise ValueError(f'Invalid number of intersections: {len(chains)}')

			obj_guess = torch.normal(0, self.init_size, lhs_2.shape, device=lhs_2.device, requires_grad=True)
			optimizer = optim.Adam([obj_guess], lr=0.1)

			prev_loss_value = 1000
			loss_value = 999
			losses = []

			with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
				i = 0
				while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
					prev_loss_value = loss_value

					score_1, factors = self.score_emb(lhs_1, rel_1, obj_guess)
					guess_regularizer = regularizer([factors[2]])
					score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess)

					atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))

					if len(chains) == 3:
						score_3, _ = self.score_emb(lhs_3, rel_3, obj_guess)
						atoms = torch.cat((atoms, torch.sigmoid(score_3)), dim=1)

					t_norm = torch.prod(atoms, dim=1)

					if disjunctive:
						t_norm = torch.sum(atoms, dim=1) - t_norm

					loss = -t_norm.mean() + guess_regularizer

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					i += 1
					bar.update(1)
					bar.set_postfix(loss=f'{loss.item():.6f}')

					loss_value = loss.item()
					losses.append(loss_value)

				if i != max_steps:

					bar.update(max_steps - i + 1)
					bar.close()
					print("Search converged early after {} iterations".format(i))

				with torch.no_grad():
					score_1 = self.forward_emb(lhs_1, rel_1)
					score_2 = self.forward_emb(lhs_2, rel_2)
					atoms = torch.stack((score_1, score_2), dim=-1)

					if disjunctive:
						atoms = torch.sigmoid(atoms)

					if len(chains) == 3:
						score_3 = self.forward_emb(lhs_3, rel_3)
						atoms = torch.cat((atoms, score_3.unsqueeze(-1)), dim=-1)

					t_norm = torch.prod(atoms, dim=-1)
					if disjunctive:
						scores = torch.sum(atoms, dim=-1) - t_norm
					else:
						scores = t_norm

		except RuntimeError as e:
			print("Cannot optimize the queries with error {}".format(str(e)))
			return None

		return scores

	def __compute_similarities__(self, x: torch.tensor):
		model_type = self.model_type().lower()
		if 'cp' in model_type:
			entities = self.rhs
		elif 'complex' in model_type:
			entities = self.embeddings[0].weight.data
		else:
			raise ValueError(f'Unknown model type {model_type}')

		scores = x @ entities.t()

		return scores

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


				#torch.cuda.empty_cache()
				gc.collect()

				#print(losses)

				if 'cp' in self.model_type().lower():
					closest_map_1, indices_rankedby_distances_1 = self.__closest_matrix__(obj_guess_1,self.rhs,similarity_metric)

					#torch.cuda.empty_cache()
					gc.collect()

					closest_map_2, indices_rankedby_distances_2 = self.__closest_matrix__(obj_guess_2,self.rhs,similarity_metric)

				elif 'complex' in self.model_type().lower():
					closest_map_1, indices_rankedby_distances_1 = self.__closest_matrix__(obj_guess_1,self.embeddings[0].weight.data,similarity_metric)
					#torch.cuda.empty_cache()
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
				#torch.cuda.empty_cache()
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

			lhs_1,rel_1,lhs_2,rel_2,lhs_3,rel_3 = self.__get_chains__(chains , graph_type  = QuerDAG.TYPE2_3.value)
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


				#torch.cuda.empty_cache()
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
			lhs_1, rel_1, rel_2, lhs_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE3_3.value)

			obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			optimizer = optim.Adam([obj_guess_1, obj_guess_2], lr=0.1)

			prev_loss_value = 1000
			loss_value = 999
			losses = []

			with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
				i = 0
				while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
					prev_loss_value = loss_value

					score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
					score_2, _ = self.score_emb(obj_guess_1, rel_2, obj_guess_2)
					score_3, factors_2 = self.score_emb(lhs_2, rel_3, obj_guess_2)
					factors = [factors_1[2], factors_2[2]]

					atoms = torch.sigmoid(torch.cat((score_1, score_2, score_3), dim=1))

					guess_regularizer = regularizer(factors)

					t_norm = torch.prod(atoms, dim=1)
					loss = -t_norm.mean() + guess_regularizer

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					i += 1
					bar.update(1)
					bar.set_postfix(loss=f'{loss.item():.6f}')

					loss_value = loss.item()
					losses.append(loss_value)

				if i != max_steps:
					bar.update(max_steps - i + 1)
					bar.close()
					print(
						"Search converged early after {} iterations".format(i))

				with torch.no_grad():
					score_2 = self.forward_emb(obj_guess_1, rel_2)
					score_3 = self.forward_emb(lhs_2, rel_3)
					atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_2), score_2, score_3), dim=-1))

					t_norm = torch.prod(atoms, dim=-1)

					scores = t_norm


		except RuntimeError as e:
			print("Cannot optimize the queries with error {}".format(str(e)))
			return None

		return scores

	def type4_3chain_optimize(self, chains: List, regularizer: Regularizer, candidates: int = 1,
							  max_steps: int = 20, step_size: float = 0.001, similarity_metric : str = 'l2', t_norm: str = 'min',
							  disjunctive=False):
		try:
			lhs_1, rel_1, lhs_2, rel_2, rel_3 = self.__get_chains__(chains, graph_type=QuerDAG.TYPE4_3.value)

			obj_guess_1 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			obj_guess_2 = torch.normal(0, self.init_size, lhs_1.shape, device=lhs_1.device, requires_grad=True)
			optimizer = optim.Adam([obj_guess_1, obj_guess_2], lr=0.1)

			prev_loss_value = 1000
			loss_value = 999
			losses = []

			with tqdm.tqdm(total=max_steps, unit='iter', disable=False) as bar:
				i = 0
				while i < max_steps and math.fabs(prev_loss_value - loss_value) > 1e-9:
					prev_loss_value = loss_value

					score_1, factors_1 = self.score_emb(lhs_1, rel_1, obj_guess_1)
					score_2, _ = self.score_emb(lhs_2, rel_2, obj_guess_1)
					score_3, factors_2 = self.score_emb(obj_guess_1, rel_3, obj_guess_2)
					factors = [factors_1[2], factors_2[2]]
					guess_regularizer = regularizer(factors)

					if not disjunctive:
						atoms = torch.sigmoid(torch.cat((score_1, score_2, score_3), dim=1))
						t_norm = torch.prod(atoms, dim=1)
					else:
						disj_atoms = torch.sigmoid(torch.cat((score_1, score_2), dim=1))
						t_conorm = torch.sum(disj_atoms, dim=1, keepdim=True) - torch.prod(disj_atoms, dim=1, keepdim=True)
						conj_atoms = torch.cat((t_conorm, torch.sigmoid(score_3)), dim=1)
						t_norm = torch.prod(conj_atoms, dim=1)

					loss = -t_norm.mean() + guess_regularizer

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()

					i += 1
					bar.update(1)
					bar.set_postfix(loss=f'{loss.item():.6f}')

					loss_value = loss.item()
					losses.append(loss_value)

				if i != max_steps:
					bar.update(max_steps - i + 1)
					bar.close()
					print(
						"Search converged early after {} iterations".format(i))

				with torch.no_grad():
					score_3 = self.forward_emb(obj_guess_1, rel_3)
					if not disjunctive:
						atoms = torch.sigmoid(torch.stack((score_1.expand_as(score_3), score_2.expand_as(score_3), score_3), dim=-1))
					else:
						atoms = torch.stack((t_conorm.expand_as(score_3), torch.sigmoid(score_3)), dim=-1)

					t_norm = torch.prod(atoms, dim=-1)
					scores = t_norm

		except RuntimeError as e:
			print("Cannot optimize the queries with error {}".format(str(e)))
			return None

		return scores

	def get_best_candidates(self,
			rel: Tensor,
			arg1: Optional[Tensor],
			arg2: Optional[Tensor],
			candidates: int = 5,
			last_step = False) -> Tuple[Tensor, Tensor]:


		z_scores, z_emb, z_indices = None, None , None

		try:
			assert (arg1 is None) ^ (arg2 is None)

			batch_size, embedding_size = rel.shape[0], rel.shape[1]

			# [B, N]

			# scores_sp = (s, p, ?)

			scores_sp, scores_po = self.candidates_score(rel, arg1, arg2)
			scores = scores_sp if arg2 is None else scores_po


			if not last_step:
				# [B, K], [B, K]
				k = min(candidates, scores.shape[1])

				z_scores, z_indices = torch.topk(scores, k=k, dim=1)
				# [B, K, E]
				z_emb = self.entity_embeddings(z_indices)
				assert z_emb.shape[0] == batch_size
				assert z_emb.shape[2] == embedding_size
			else:
				z_scores = scores

				z_indices = torch.arange(z_scores.shape[1]).view(1,-1).repeat(z_scores.shape[0],1).to(Device)
				z_emb = self.entity_embeddings(z_indices)

				# z_scores, z_indices = torch.topk(scores, k=scores.shape[1], dim=1)
				# # z_scores = scores
				# # [B, K, E]
				# z_emb = self.entity_embeddings(z_indices)
				# assert z_emb.shape[0] == batch_size
				# assert z_emb.shape[2] == embedding_size



			del z_indices
			#torch.cuda.empty_cache()

		except RuntimeError as e:
			print("Cannot find the candidates with error: ", e)
			return z_scores, z_emb


		return z_scores, z_emb


	def t_norm(self,
			tens_1: Tensor,
			tens_2: Tensor,
			t_norm: str = 'min',
			) -> Tensor:
			if 'min' in t_norm:
				return torch.min(tens_1, tens_2)
			elif 'prod' in t_norm:
				return tens_1 * tens_2

	def t_conorm(self,
			tens_1: Tensor,
			tens_2: Tensor,
			t_conorm: str = 'max',
			) -> Tensor:
			if 'max' in t_conorm:
				return torch.max(tens_1, tens_2)
			elif 'prod' in t_conorm:
				return (tens_1+tens_2) - (tens_1 * tens_2)

	def min_max_rescale(self,x):
		return (x-torch.min(x))/(torch.max(x)- torch.min(x))

	def query_answering_BF(self, env: DynKBCSingleton ,  regularizer: Regularizer, candidates: int = 5,\
							similarity_metric : str = 'l2', t_norm: str = 'min' , batch_size = 4):


		res = None
		try:

			if 'disj' in env.graph_type:
				objective = self.t_conorm
			else:
				objective = self.t_norm

			chains, chain_instructions = env.chains, env.chain_instructions

			nb_queries, embedding_size = chains[0][0].shape[0], chains[0][0].shape[1]


			scores = None

			# data_loader = DataLoader(dataset=chains, batch_size=16, shuffle=False)

			batches = make_batches(nb_queries, batch_size)

			for batch in tqdm.tqdm(batches):

				nb_branches = 1
				nb_ent = 0
				batch_scores = None
				candidate_cache = {}

				batch_size = batch[1] - batch[0]
				#torch.cuda.empty_cache()

				for inst_ind, inst in enumerate(chain_instructions):
					with torch.no_grad():
						if 'hop' in inst:

							ind_1 = int(inst.split("_")[-2])
							ind_2 = int(inst.split("_")[-1])


							last_hop = False
							for ind in [ind_1, ind_2]:

								last_step =  (inst_ind == len(chain_instructions)-1) and last_hop

								lhs,rel,rhs = chains[ind]


								# [a, p, X], [X, p, Y][Y, p, Z]

								if lhs is not None:
									lhs = lhs[batch[0]:batch[1]]
								else:
									batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
									lhs = lhs_3d.view(-1, embedding_size)

								rel = rel[batch[0]:batch[1]]
								rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
								rel = rel.view(-1, embedding_size)


								if f"rhs_{ind}" not in candidate_cache:
									z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step)

									# [Num_queries * Candidates^K]
									z_scores_1d = z_scores.view(-1)
									if 'disj' in env.graph_type:
										z_scores_1d = torch.sigmoid(z_scores_1d)

									# B * S
									nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
									nb_branches = nb_sources // batch_size
									if not last_step:
										batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
									else:
										nb_ent = rhs_3d.shape[1]
										batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)


									candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)
									candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)

								else:
									batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
									candidate_cache[f"lhs_{ind+1}"] = (batch_scores, rhs_3d)
									last_hop =  True
									del lhs, rel
									# #torch.cuda.empty_cache()
									continue


								last_hop =  True
								del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
								# #torch.cuda.empty_cache()

						elif 'inter' in inst:
							ind_1 = int(inst.split("_")[-2])
							ind_2 = int(inst.split("_")[-1])

							indices = [ind_1, ind_2]

							if len(inst.split("_")) > 3:
								ind_1 = int(inst.split("_")[-3])
								ind_2 = int(inst.split("_")[-2])
								ind_3 = int(inst.split("_")[-1])

								indices = [ind_1, ind_2, ind_3]

							for interesction_num, ind in enumerate(indices):

								last_step =  (inst_ind == len(chain_instructions)-1) and ind == indices[0]

								lhs,rel,rhs = chains[ind]

								if lhs is not None:
									lhs = lhs[batch[0]:batch[1]]
									lhs = lhs.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
									lhs = lhs.view(-1, embedding_size)

								else:
									batch_scores, lhs_3d = candidate_cache[f"lhs_{ind}"]
									lhs = lhs_3d.view(-1, embedding_size)
									nb_sources = lhs_3d.shape[0]*lhs_3d.shape[1]
									nb_branches = nb_sources // batch_size

								rel = rel[batch[0]:batch[1]]
								rel = rel.view(-1, 1, embedding_size).repeat(1, nb_branches, 1)
								rel = rel.view(-1, embedding_size)

								if interesction_num > 1:
									batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
									rhs = rhs_3d.view(-1, embedding_size)
									z_scores = self.score_fixed(rel, lhs, rhs, candidates)

									z_scores_1d = z_scores.view(-1)
									if 'disj' in env.graph_type:
										z_scores_1d = torch.sigmoid(z_scores_1d)

									batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores, t_norm)

									continue


								if f"rhs_{ind}" not in candidate_cache or last_step:
									z_scores, rhs_3d = self.get_best_candidates(rel, lhs, None, candidates, last_step)

									# [B * Candidates^K] or [B, S-1, N]
									z_scores_1d = z_scores.view(-1)
									# print(z_scores_1d)
									if 'disj' in env.graph_type:
										z_scores_1d = torch.sigmoid(z_scores_1d)

									nb_sources = rhs_3d.shape[0]*rhs_3d.shape[1]
									nb_branches = nb_sources // batch_size

									if not last_step:
										batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, candidates).view(-1), t_norm)
									else:
										nb_ent = rhs_3d.shape[1]
										batch_scores = z_scores_1d if batch_scores is None else objective(z_scores_1d, batch_scores.view(-1, 1).repeat(1, nb_ent).view(-1), t_norm)

									candidate_cache[f"rhs_{ind}"] = (batch_scores, rhs_3d)

									if last_step:
										candidate_cache[f"rhs_{ind+1}"] = (batch_scores, rhs_3d)

								else:
									batch_scores, rhs_3d = candidate_cache[f"rhs_{ind}"]
									candidate_cache[f"rhs_{ind+1}"] = (batch_scores, rhs_3d)

									last_hop =  True
									del lhs, rel
									#torch.cuda.empty_cache()
									continue

								del lhs, rel, rhs, rhs_3d, z_scores_1d, z_scores
								#torch.cuda.empty_cache()


				if batch_scores is not None:
					# [B * entites * S ]
					# S ==  K**(V-1)
					scores_2d = batch_scores.view(batch_size,-1, nb_ent )
					res, _ = torch.max(scores_2d, dim=1)
					scores = res if scores is None else torch.cat([scores,res])

					# candidate_cache.clear()
					#torch.cuda.empty_cache()
					del batch_scores, scores_2d, res,candidate_cache
					#torch.cuda.empty_cache()
					gc.collect()


				else:
					print("Batch Scores are empty: an error went uncaught.")
					print(traceback.print_exc())
					pass

				res = scores
					#torch.cuda.empty_cache()


		except RuntimeError as e:
			print(traceback.print_exc())
			print("Cannot complete iterative BF with error: ",e)
			return res
		return res


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
			x_norm = (torch.pow(x, 2)).sum(1).view(-1, 1)
			if y is not None:
				y_norm = (torch.pow(y, 2)).sum(1).view(1, -1)
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



	def entity_embeddings(self, indices: Tensor):
		return self.rhs(indices)

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


		if torch.sum(queries[:, 0]).item() > 0:
			lhs = self.lhs(queries[:, 0])
		else:
			lhs = None

		if torch.sum(queries[:, 1]).item() > 0:
			rel = self.rel(queries[:, 1])
		else:
			rel = None

		if torch.sum(queries[:, 2]).item() > 0:
			rhs = self.rhs(queries[:, 2])
		else:
			rhs = None

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

		self.init_size = init_size

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


	def entity_embeddings(self, indcies: Tensor):
		return self.embeddings[0](indcies)


	def score_fixed(self,
			  rel: Tensor,
			  arg1: Tensor,
			  arg2: Tensor,
			  *args, **kwargs) -> Tensor:
		# [B, E]
		rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
		arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]
		arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

		# [B] Tensor
		score1 = torch.sum(rel_real * arg1_real * arg2_real, 1)
		score2 = torch.sum(rel_real * arg1_img * arg2_img, 1)
		score3 = torch.sum(rel_img * arg1_real * arg2_img, 1)
		score4 = torch.sum(rel_img * arg1_img * arg2_real, 1)

		res = score1 + score2 + score3 - score4

		del score1,score2, score3, score4, rel_real, rel_img, arg1_real, arg1_img, arg2_real, arg2_img
		#torch.cuda.empty_cache()
		# [B] Tensor
		return res

	def candidates_score(self,
				rel: Tensor,
				arg1: Optional[Tensor],
				arg2: Optional[Tensor],
				*args, **kwargs) -> Tuple[Optional[Tensor], Optional[Tensor]]:


		emb = self.embeddings[0].weight

		rel_real, rel_img = rel[:, :self.rank], rel[:, self.rank:]
		emb_real, emb_img = emb[:, :self.rank], emb[:, self.rank:]

		# [B] Tensor

		score_sp = score_po = None

		if arg1 is not None:
			arg1_real, arg1_img = arg1[:, :self.rank], arg1[:, self.rank:]

			score1_sp = (rel_real * arg1_real) @ emb_real.t()
			score2_sp = (rel_real * arg1_img) @ emb_img.t()
			score3_sp = (rel_img * arg1_real) @ emb_img.t()
			score4_sp = (rel_img * arg1_img) @ emb_real.t()

			score_sp = score1_sp + score2_sp + score3_sp - score4_sp

		if arg2 is not None:
			arg2_real, arg2_img = arg2[:, :self.rank], arg2[:, self.rank:]

			score1_po = (rel_real * arg2_real) @ emb_real.t()
			score2_po = (rel_real * arg2_img) @ emb_img.t()
			score3_po = (rel_img * arg2_img) @ emb_real.t()
			score4_po = (rel_img * arg2_real) @ emb_img.t()

			score_po = score1_po + score2_po + score3_po - score4_po

		return score_sp, score_po

	def score_emb(self, lhs_emb, rel_emb, rhs_emb):
		lhs = lhs_emb[:, :self.rank], lhs_emb[:, self.rank:]
		rel = rel_emb[:, :self.rank], rel_emb[:, self.rank:]
		rhs = rhs_emb[:, :self.rank], rhs_emb[:, self.rank:]

		return torch.sum(
			(lhs[0] * rel[0] - lhs[1] * rel[1]) * rhs[0] +
			(lhs[0] * rel[1] + lhs[1] * rel[0]) * rhs[1],
			1, keepdim=True), (
			torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
			torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
			torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
		)

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

	def forward_emb(self, lhs, rel):
		lhs = lhs[:, :self.rank], lhs[:, self.rank:]
		rel = rel[:, :self.rank], rel[:, self.rank:]

		to_score = self.embeddings[0].weight
		to_score = to_score[:, :self.rank], to_score[:, self.rank:]
		return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
				(lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

	def get_rhs(self, chunk_begin: int, chunk_size: int):
		return self.embeddings[0].weight.data[
			chunk_begin:chunk_begin + chunk_size
		].transpose(0, 1)

	def get_queries_separated(self, queries: torch.Tensor):
		lhs = self.embeddings[0](queries[:, 0])
		rel = self.embeddings[1](queries[:, 1])

		return (lhs, rel)

	def get_full_embeddigns(self, queries: torch.Tensor):

		if torch.sum(queries[:, 0]).item() > 0:
			lhs = self.embeddings[0](queries[:, 0])
		else:
			lhs = None

		if torch.sum(queries[:, 1]).item() > 0:

			rel = self.embeddings[1](queries[:, 1])
		else:
			rel = None

		if torch.sum(queries[:, 2]).item() > 0:
			rhs = self.embeddings[0](queries[:, 2])
		else:
			rhs = None

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
