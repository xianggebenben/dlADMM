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

@inproceedings{Wang:2019:AED:3292500.3330936,

 author = {Wang, Junxiang and Yu, Fuxun and Chen, Xiang and Zhao, Liang},
 
 title = {ADMM for Efficient Deep Learning with Global Convergence},
 
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
 
 series = {KDD '19},
 
 year = {2019},
 
 isbn = {978-1-4503-6201-6},
 
 location = {Anchorage, AK, USA},
 
 pages = {111--119},
 
 numpages = {9},
 
 url = {http://doi.acm.org/10.1145/3292500.3330936},
 
 doi = {10.1145/3292500.3330936},
 
 acmid = {3330936},
 
 publisher = {ACM},
 
 address = {New York, NY, USA},
 
 keywords = {alternating direction method of multipliers, deep learning, global convergence},
 
} 
