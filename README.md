# Query Answering In Knowledge Space

This Code is the implementation of "Query Answering with Knowledge Graph
Embeddings via Continuous Optimisation" inspired from ["Querying Complex Networks in Vector Space"](https://github.com/williamleif/graphqembed)

The implementation builds upon and expands the code-base from [Canonical Tensor Decomposition for Knowledge Base Completion](https://arxiv.org/abs/1806.07297) from  FAIR.

____

## Setup

The Setup process is similar the the original [repo](https://github.com/facebookresearch/kbc) as the process for installation has been altered sloghtly to accomodate fror the expanded functional

### 1) Create an environemnt and install the package and the requirements

```
conda create --name kbc_env python=3.7
source activate kbc_env
conda install --file requirements.txt -c pytorch
pip install requirements.txt
python setup.py install
```

Here conda environments can be replaced manual <i>virtualenv</i> environments.

### 2a) Download the data used in original KBC

```
chmod +x download_data.sh
./download_data.sh
```

### 2b) Download the Bio Data used as the benchmark for "Querying Complex Networks in Vector Space"

The Bio data can be downloaded [here](https://snap.stanford.edu/nqe/bio_data.zip)

Unzip the data under the name "Bio" in the folder "/Query-Answering-In-Knowledge-Space/kbc/src_data"

### 2c) Convert and Process the Bio data to appropriate form.

```
python kbc/bio_data_process.py
```

### 3) Process all of the datasets to a Knowledge Graph format.

```
python kbc/process_datasets.py
```

## Training

You can train a KBC model on any one of the datasets processed in the previous step

Example:
```
python kbc/learn.py --dataset Bio --model ComplEx --max_epochs 30 \
--model_save_schedule 10 --valid 10 --reg 5e-2 --batch_size 1024 --rank 128

```

All of the input parameters and options can be found in both the code documentation or simply by using `-help` in the command line.

## Optimization and Benchmarks

### Sampling chains

To Sample longer chains of different types as described in the paper(TODO: Add link), use

Example:
```
python kbc/chain_dataset.py --dataset Bio --threshold 5000
```

### Optimizing

Find the target vectors and answer the queries as described in the paper(TODO: Add link). (Reproduce the benchmarks this way)

```
python kbc/query_space_optimize.py --model_path models/Bio-model-epoch-30-1566308599.pt \
--dataset Bio --dataset_mode test --similarity_metric l2 --chain_type 1_3


```

### Results

|         |    Bio   |                                                            |   |                            WN18                            |
|:-------:|:--------:|:----------------------------------------------------------:|:-:|:----------------------------------------------------------:|
|         |    GQE   | Continuous Optimization with KG embeddings. (Our solution) |   | Continuous Optimization with KG embeddings. (Our solution) |
|  Type-1 |   0.99   |                            0.99                            |   |                          0.987742                          |
| Type1-2 | 0.931927 |                        $\sim$0.95737                       |   |                          0.996718                          |
| Type2-2 | 0.925571 |                        $\sim$0.96639                       |   |                          0.996269                          |
| Type1-3 |  0.89373 |                        $\sim$0.96461                       |   |                          0.9985055                         |
| Type2-3 | 0.881848 |                        $\sim$0.93691                       |   |                          0.9907173                         |
| Type3-3 |  0.87890 |                        $\sim$0.93952                       |   |                         0.99603959                         |
| Type4-3 | 0.886478 |                        $\sim$0.85132                       |   |                         0.99331019                         |


|          | Hits@1 | Hits@1 |
|:--------:|--------|--------|
|          |   Bio  |  WN18  |
|  Type 1  |  0.561 | 0.9774 |
| Type 1-2 |  0.704 | 0.8642 |
| Type 2-2 |  0.599 | 0.9336 |
| Type 1-3 |  0.629 | 0.9312 |
| Type 2-3 |  0.599 | 0.8868 |
| Type 3-3 |  0.582 | 0.8112 |
| Type 4-3 |  0.503 | 0.8176 |


___

## TODO

- [ ] Refactor the code into more general functions for different chains types in ```models.py``` and ```query_space_optimize.py```
