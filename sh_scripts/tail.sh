# taskset -c 20-25 python3 experiments/tail.py --n=5 --nrep=2 --d=2
for d in 1 10 20 50
do
    taskset -c 20-25 python3 experiments/tail.py --n=500 --nrep=50 --d=$d &
done
wait