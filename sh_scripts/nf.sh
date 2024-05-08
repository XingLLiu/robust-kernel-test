# for i in $(seq 5 9) # $(seq 0 4)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed &
# done
# wait

# generate parametric boot samples
for i in $(seq 0 19)
do
    seed=$(($i))
    c1=$(($i))
    c2=$(($i))
    # taskset -c $c1-$c2 python3 experiments/nf.py --n=3 --nrep=2 --seed=$seed --param_boot=True &
    taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=25 --seed=$seed --param_boot=True #&
done
wait


# taskset -c 0-2 python3 experiments/nf.py --n=3 --nrep=2 --seed=0
