# taskset -c 20-25 python3 experiments/rate.py --nrep=100 --r=0.6 --gen=True
# taskset -c 20-25 python3 experiments/rate.py --nrep=100 --r=0.5 --gen=True
# taskset -c 20-25 python3 experiments/rate.py --nrep=100 --r=0.4 --gen=True
for r in 0.3 0.4 0.45 0.5 0.55 0.6
do
    taskset -c 20-25 python3 experiments/rate.py --nrep=400 --r=$r --gen=True &
done
wait