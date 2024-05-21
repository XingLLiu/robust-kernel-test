for n in 100 # 100 200 300 400 500
do
    for d in 1 100 #1 5 10 20 50 100
    do
        echo "Running n=$n d=$d"
        # taskset -c 0-20 python3 experiments/ms.py --n=$n --d=$d --nrep=50 --gen=True
        taskset -c 0-20 python3 experiments/ms.py --n=$n --d=$d --nrep=20 --gen=False &
    done
    wait
done