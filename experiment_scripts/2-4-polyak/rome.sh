#!/usr/bin/env bash
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.5 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.75 & --seed 10
wait $!
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.875 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 256 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.01 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 & --seed 10
mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25 --network_class FourierMLP --sigma 0.001 --fourier_dim 1024 --concatenate_fourier --train_B --n_hidden 2 --polyak 0.9 & --seed 10

