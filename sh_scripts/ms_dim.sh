for d in 1 5 10 20 50 100
do
    echo "Running d=$d"
    taskset -c 0-30 python3 experiments/ms.py --n=500 --d=$d --nrep=20
done