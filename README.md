# dlADMM: Deep Learning Optimization via Alternating Direction Method of Multipliers
This is a  implementation of deep learning Alternating Direction Method of Multipliers(dlADMM) for the task of fully-connected neural network
problem, as described in our paper:

Junxiang Wang, Fuxun Yu, Xiang Chen, and Liang Zhao. [ADMM for Efficient Deep Learning with Global Convergence.](https://arxiv.org/abs/1905.13611) (KDD 2019)

## Installation

python setup.py install

## Requirements

cupy-cuda90(>=6.0.0 is recommended)

tensorflow

keras

## Run the Demo

python main.py

## Data

Two benchmark datasets MNIST and Fashion-MNIST are included in this package.

## Cite

Please cite our paper if you use this code in your own work:

@inproceedings{10.1145/3292500.3330936,
author = {Wang, Junxiang and Yu, Fuxun and Chen, Xiang and Zhao, Liang},
title = {ADMM for Efficient Deep Learning with Global Convergence},
year = {2019},
isbn = {9781450362016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3292500.3330936},
doi = {10.1145/3292500.3330936},
booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
pages = {111–119},
numpages = {9},
keywords = {alternating direction method of multipliers, deep learning, global convergence},
location = {Anchorage, AK, USA},
series = {KDD ’19}
}
