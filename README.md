2021.12 Update: The pytorch version of the dlADMM code for the GCN model are available.
-------------------------------------------------------------------------------------------------------------------------
2021.6 Update: The pytorch and the tensorflow versions of the dlADMM code for the MLP model are available.
-------------------------------------------------------------------------------------------------------------------------
2020.4 Update: The dlADMM code is optimized without the cupy library and runs 10 times faster than the previous version. Each iteration takes only several seconds.
-------------------------------------------------------------------------------------------------------------------------

# dlADMM: Deep Learning Optimization via Alternating Direction Method of Multipliers
This is a  implementation of deep learning Alternating Direction Method of Multipliers(dlADMM) for the task of fully-connected neural network
problem, as described in our paper:

Junxiang Wang, Fuxun Yu, Xiang Chen, and Liang Zhao. [ADMM for Efficient Deep Learning with Global Convergence.](https://arxiv.org/abs/1905.13611) (KDD 2019)

## Installation
python setup.py install

## Requirements
GCN

For the pytorch version:

torch==1.8.1

numpy==1.21.2

MLP

For the tensorflow version:

tensorflow==2.4.1

numpy==1.16.4

For the pytorch version:

torch==1.8.1

numpy==1.16.4


## Run the Demo

python main.py

## Data
For the MLP model, two benchmark datasets MNIST and Fashion-MNIST are included in this package.

For the GCN model, five benchmark datasets Cora, PubMed, Citeseer, Coauthor-CS and Coauthor-Physics are included in this package.

## Cite
Please cite our following paper if you use our code in your own work:

@inproceedings{wang2019admm,

author = {Wang, Junxiang and Yu, Fuxun and Chen, Xiang and Zhao, Liang},

title = {ADMM for Efficient Deep Learning with Global Convergence},

year = {2019},

isbn = {9781450362016},

publisher = {Association for Computing Machinery},

address = {New York, NY, USA},

doi = {10.1145/3292500.3330936},

booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},

numpages = {9},

keywords = {alternating direction method of multipliers, deep learning, global convergence},

location = {Anchorage, AK, USA},

series = {KDD ’19}

}

The previous paper on training neural networks via ADMM "Training Neural Networks Without Gradients:
A Scalable ADMM Approach" has published their code at https://gitlab.umiacs.umd.edu/tomg/admm_nets.
