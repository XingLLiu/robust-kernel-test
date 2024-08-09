taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=100 --data=galaxy --exp=level --gen=True
taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=100 --data=galaxy --exp=power --gen=True
