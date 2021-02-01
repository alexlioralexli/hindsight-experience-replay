total = 0
envs = ['mpirun -np 8 python -u train.py --env-name FetchPush-v1']

for env in envs:
    commands = [env]
    for type in ['--train_B', '--concatenate_fourier --train_B']:
        for sigma in [0.01, 0.001]:
            commands.append(env + f' --network_class FourierMLP --sigma {sigma} --fourier_dim 1024 {type}')
    count = 0
    for command in commands:
        # gpus = list(range(8,10))
        gpus = list(range(10))
        for seed in [10, 20]:
            if total % 8 == 0:
                print(total)
            total += 1
            print(f'{command} --seed {seed} &')
            count = (count + 1) % len(gpus)

