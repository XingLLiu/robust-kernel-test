# taskset -c 0-20 python3 experiments/ms.py --n=3 --d=2 --nrep=2 --gen=True

for n in 500
do
    for d in 1 10 50 100
    do
        echo "Running n=$n d=$d"
        # taskset -c 0-20 python3 experiments/ms.py --n=$n --d=$d --nrep=50 --gen=True
        taskset -c 0-20 python3 experiments/ms.py --n=$n --d=$d --nrep=50 --gen=True &
    done
    wait
done