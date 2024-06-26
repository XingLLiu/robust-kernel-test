# taskset -c 20-25 python3 experiments/mixture.py --n=5 --nrep=2 --nmix=5 --d=1
for d in 2 #10 50
do
    taskset -c 20-25 python3 experiments/mixture.py --n=1000 --nrep=50 --nmix=5 --d=$d #&
done
wait