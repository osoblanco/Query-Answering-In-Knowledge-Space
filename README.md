# Query Answering In Knowledge Space

This Code is the implementation of "Optimized Query Answering in Knowledge Space" inspired from ["Querying Complex Networks in Vector Space"](https://github.com/williamleif/graphqembed)

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

All of the input parametrsoptions can be found in both the code documentation or simply by using `-help` in the command line.

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
--dataset Bio --dataset_mode test --similarity_metric l2

```
