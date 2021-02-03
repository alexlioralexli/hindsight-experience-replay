#!/usr/bin/env bash
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 10 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 20 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 30 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 40 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 50 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --seed 50 &