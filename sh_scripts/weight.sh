# taskset -c 20-25 python3 experiments/weight.py --n=5 --nrep=2 --dim=50 --gen=True
# taskset -c 20-25 python3 experiments/weight.py --n=500 --nrep=50 --d=50 --gen=True
taskset -c 20-25 python3 experiments/weight.py --n=500 --nrep=50 --d=1 --gen=True