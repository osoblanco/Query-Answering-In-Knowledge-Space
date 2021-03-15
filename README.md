# Complex Query Decomposition

This repository contains the official implementation for our ICLR 2021 paper, [**Complex Query Answering with Neural Link Predictors**](https://openreview.net/forum?id=Mos9F9kDwkz):

```bibtex
@inproceedings{
arakelyan2021complex,
title={Complex Query Answering with Neural Link Predictors},
author={Erik Arakelyan and Daniel Daza and Pasquale Minervini and Michael Cochez},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=Mos9F9kDwkz}
}
```

In this work we present CQD, a method that reuses a pretrained link predictor to answer complex queries, by scoring atom predicates independently and aggregating the scores via t-norms and t-conorms.

Our code is based on the implementation of ComplEx-N3 available [here](https://github.com/facebookresearch/kbc).

Please follow the instructions next to reproduce the results in our experiments.

### 1. Install the requirements

We recommend creating a new environment:

```
conda create --name cqd python=3.8 && conda activate cqd
pip install -r requirements.txt
```

### 2. Download the data

We use 3 knowledge graphs: FB15k, FB15k-237, and NELL.
From the root of the repository, download and extract the files to obtain the folder `data`, containing the sets of triples and queries for each graph.

```sh
wget http://data.neuralnoise.com/cqd-data.tgz
tar xvf cqd-data.tgz
```

### 3. Download the models

Then you need neural link prediction models -- one for each of the datasets.
Our pre-trained neural link prediction models are available here:

```sh
wget http://data.neuralnoise.com/cqd-models.tgz
tar xvf cqd-data.tgz
```

### 3. Alternative -- Train your own models

To obtain entity and relation embeddings, we use ComplEx. Use the next commands to train the embeddings for each dataset.

#### FB15k

```sh
python -m kbc.learn data/FB15k --rank 1000 --reg 0.01 --max_epochs 100  --batch_size 100
```

#### FB15k-237

```sh
python -m kbc.learn data/FB15k-237 --rank 1000 --reg 0.05 --max_epochs 100  --batch_size 1000
```

#### NELL

```sh
python -m kbc.learn data/NELL --rank 1000 --reg 0.05 --max_epochs 100  --batch_size 1000
```

Once training is done, the models will be saved in the `models` directory.

### 4. Answering queries with CQD

CQD can answer complex queries via continuous (CQD-CO) or combinatorial optimisation (CQD-Beam).

#### CQD-CO

Use the `kbc.cqd_co` script to answer queries, providing the path to the dataset, and the saved link predictor trained in the previous step. For example,

```sh
python -m kbc.cqd_co data/FB15k --model_path models/[model_filename].pt --chain_type 1_2
```

#### CQD-Beam

For this variant, use `kbc.cqd_beam` similarly as in CQD-CO.

```sh
python -m kbc.cqd_beam --model_path models/[model_filename].pt
```
