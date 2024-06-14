### Data ###
# # generate test samples
# for i in $(seq 5 9) # $(seq 0 4)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed &
# done
# wait

# # generate mnist samples
# for i in $(seq 0 4)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed --data=mnist &
# done
# wait

# for i in $(seq 5 9)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed --data=mnist &
# done
# wait

# # generate full-noise test samples
# for i in $(seq 0 4)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed --data=model_full &
# done
# wait

# for i in $(seq 5 9)
# do
#     seed=$(($i * 9))
#     c1=$(($i * 2))
#     c2=$(($i * 2 + 1))
#     taskset -c $c1-$c2 python3 experiments/nf.py --n=500 --nrep=2 --seed=$seed --data=model_full &
# done
# wait


### Test ###
# # level
# taskset -c 0-2 python3 experiments/nf_test.py --n=500 --nexp=500 --setup=level

# # level with full-noise
# date > test.txt
# for n in 100 200 300 400 500
# do
#     taskset -c 0-20 python3 experiments/nf_test.py --n=500 --nexp=$n --setup=model_full
#     echo "Finished n=$n" >> test.txt
# done
# echo "Finished testing NF full noise" >> test.txt
# date >> test.txt

date > test.txt
taskset -c 0-20 python3 experiments/nf_test.py --n=500 --nexp=500 --setup=model_full
echo "Finished testing NF full noise" >> test.txt
date >> test.txt

# # power
# taskset -c 3-10 python3 experiments/nf_test.py --n=500 --nexp=200 --setup=power &
# taskset -c 11-18 python3 experiments/nf_test.py --n=500 --nexp=500 --setup=power

# # mnist
# for nexp in 100 200 300 400 500
# do
#     taskset -c 0-10 python3 experiments/nf_test.py --n=500 --nexp=$nexp --setup=mnist
# done

