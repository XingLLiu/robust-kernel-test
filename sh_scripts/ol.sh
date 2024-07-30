# taskset -c 20-25 python3 experiments/ol.py --n=500 --nrep=50 --gen=True --run_lq=True
# taskset -c 20-25 python3 experiments/ol.py --n=500 --gen=True --nrep=100
taskset -c 20-25 python3 experiments/ol.py --n=500 --nrep=100 --wild=True
