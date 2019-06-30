# deep learning Alternating Direction Method of Multipliers
This is a  implementation of deep learning Alternating Direction Method of Multipliers(dlADMM) for the task of fully-connected neural network
problem, as described in our paper:

Junxiang Wang, Fuxun Yu, Xiang Chen, and Liang Zhao. ADMM for Efficient Deep Learning with Global Convergence (KDD 2019)

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

@article{wang2019admm,

  title={ADMM for Efficient Deep Learning with Global Convergence},
  
  author={Wang, Junxiang and Yu, Fuxun and Chen, Xiang and Zhao, Liang},
  
  journal={arXiv preprint arXiv:1905.13611},
  
  year={2019}
  
}
