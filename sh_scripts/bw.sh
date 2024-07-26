# taskset -c 20-25 python3 experiments/bw.py --n=5 --nrep=2 --gen=True --exp=ol
taskset -c 20-25 python3 experiments/bw.py --n=100 --nrep=100 --d=50 --gen=True --exp=eps
# taskset -c 20-25 python3 experiments/bw.py --n=100 --nrep=50 --d=10 --gen=True --exp=eps