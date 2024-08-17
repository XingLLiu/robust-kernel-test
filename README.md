# Code for Robust-KSD Test
This repository contains the code for reproducing the experiments in the paper 
- [Liu, X. and Briol, F.X., 2024. On the Robustness of Kernel Goodness-of-Fit Tests. arXiv preprint arXiv:2408.05854.](https://arxiv.org/abs/2408.05854)

# How to install?
## Install as a package
Before running any scripts, run the following to install the current package and the dependencies. 
```
pip install git+https://github.com/XingLLiu/robust-kernel-test
```

## Install dependencies only
Alternatively, to install only the dependencies but not as a package, run
```
pip install -r requirements.txt
```

# Examples
After installing as a package, it can be loaded as a Python module using
```
import rksd
```