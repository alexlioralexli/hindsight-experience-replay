#!/usr/bin/env bash
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.5 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.5 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.5 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.5 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 --seed 50 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.75 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.75 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.75 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.75 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 --seed 50 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.875 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.875 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.875 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.875 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 --seed 50 &
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.9 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.9 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.9 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class MLP --n_hidden 3 --polyak 0.9 --seed 50 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 --seed 20 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 --seed 30 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 --seed 40 &
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 --seed 50 &
