total = 0
envs = ['mpirun -np 8 python -u train.py --env-name FetchPush-v1 --n-epochs 25',
        'mpirun -np 8 python -u train.py --env-name FetchSlide-v1 --n-epochs=100']

for env in envs:
    commands = []
    for type in ['--train_B', '--concatenate_fourier --train_B']:
        for fourier_dim in [256, 1024]:
            for sigma in [0.01, 0.001]:
                commands.append(env + f' --network_class FourierMLP --sigma {sigma} --fourier_dim {fourier_dim} {type} --n_hidden 2')
    count = 0
    for command in commands:
        for seed in [10]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'{command} --seed {seed}')

