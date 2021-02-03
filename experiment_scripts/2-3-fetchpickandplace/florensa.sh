#!/usr/bin/env bash
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 16 python -u train.py --env-name FetchPickAndPlace-v1 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
