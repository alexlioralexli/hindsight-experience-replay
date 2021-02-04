total = 0
envs = ['mpirun -np 1 python -u train.py --env-name FetchReach-v1 --n-cycles 1 --n-epochs 25']

for env in envs:
    commands = []
    for polyak in [0.5, 0.75, 0.875, 0.9]:
        for type in ['--concatenate_fourier --train_B']:
            for fourier_dim in [256, 1024]:
                for sigma in [0.01, 0.001]:
                    commands.append(env + f' --network_class FourierMLP --sigma {sigma} --fourier_dim {fourier_dim} {type} --n_hidden 2 --polyak {polyak} &')
    count = 0
    for command in commands:
        for seed in [10]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'{command} --seed {seed}')

