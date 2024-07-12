# taskset -c 0-20 python3 experiments/rbm.py --seed=2024 --nrep=50 --n=500 --dim=50 --hdim=10 --gen=True
taskset -c 0-20 python3 experiments/rbm.py --seed=2024 --nrep=50 --n=500 --dim=50 --hdim=10

# taskset -c 0-20 python3 experiments/rbm.py --seed=2024 --nrep=50 --n=1000 --dim=50 --hdim=10 --gen=True
# taskset -c 0-20 python3 experiments/rbm.py --seed=2024 --nrep=2 --n=5 --dim=50 --hdim=10 --gen=True
