# taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=100 --data=galaxy --gen=True --exp=level
# taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=100 --data=galaxy --gen=True --exp=power

taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=100 --data=galaxy --gen=True --exp=level
# taskset -c 0-20 python3 experiments/kef.py --seed=2024 --nrep=50 --data=galaxy --gen=True --exp=power
