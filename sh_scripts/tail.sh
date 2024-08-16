# taskset -c 20-25 python3 experiments/tail.py --n=5 --nrep=2 --d=2
taskset -c 20-25 python3 experiments/tail.py --n=500 --nrep=100 --d=1 --gen=True