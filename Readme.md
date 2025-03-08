# Sampling in High-Dimensions using Stochastic Interpolants and Forward-Backward Stochastic Differential Equations

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

Official code for the paper:

**Sampling in High-Dimensions using Stochastic Interpolants and Forward-Backward Stochastic Differential Equations**  
*Anand Jerry George and Nicolas Macris*  
Accepted to AISTATS, 2025  

[[Paper]]() [[ArXiv]](https://arxiv.org/abs/2502.00355)

## Overview
This repository contains the code implementation of the methods and experiments presented in our paper. The goal of this work is to approach the sampling problem using Stochastic interpolants framework.

The python packages required to run this code can be found in `requirements.txt`. The main code files can be found in the folder `stint_sampler`.

### Training the Model
Run the `main.py` script for training the model.

### Hyperparameters
Hyperparameters used in the implementation can be found in the `configs` directory.

### Evaluating the Model
The script `make_plots.py` can be used to generate the figures in the paper.

## Citation
If you find this code useful, please consider citing our paper.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
