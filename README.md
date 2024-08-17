# Code for Robust-KSD Test
This repository contains the code for reproducing the experiments in the paper 
- [Liu, X. and Briol, F.X., 2024. On the Robustness of Kernel Goodness-of-Fit Tests. arXiv preprint arXiv:2408.05854.](https://arxiv.org/abs/2408.05854)

# How to install?
## Install as a package
Before running any scripts, run the following to install the current package and the dependencies. 
```bash
pip install git+https://github.com/XingLLiu/robust-kernel-test
```

## Install dependencies only
Alternatively, to install only the dependencies but not as a package, run
```bash
pip install -r requirements.txt
```

# Examples
After installing as a package, it can be loaded as a Python module using
```python
import rksd
```

# Folder structure
```bash
.
├── rksd                          # Source files for robust-KSD test and benchmarks
├── sh_scripts                    # Shell scripts to run experiments
├── res                           # Folder to store results
├── figs                          # Folder to store figures
├── experiments                   # Scripts for experiments
├── setup.py                      # Setup file for easy-install of rksd
└── README.md
```