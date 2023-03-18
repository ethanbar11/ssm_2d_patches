dataset = "CIFAR100"
# dataset = "T-IMNET"
command = "CUDA_VISIBLE_DEVICES={i} python main.py --model {model} --dataset {dataset} " \
          "--project {dataset}_for_rebuttal_with_different_seeds --name {model_name},seed={seed} --seed {seed} --ema {ema}"

vit_command = "CUDA_VISIBLE_DEVICES={i} python main.py --model {model} --dataset {dataset} " \
              "--project {dataset}_for_rebuttal_with_different_seeds --name {model_name}_og={seed} --seed {seed}"

models = {'vit': ['None'], 'mega': ['none', 'ssm_2d', 'ema']}
seed_amount = 3
counter = 0
for model_name, ema_options in models.items():
    for option in ema_options:
        for seed in range(seed_amount):
            run_name = f"{model_name}"
            if model_name == 'mega':
                run_name += f"_{option}"
            command_to_run = command.format(i=counter, model=model_name, dataset=dataset, model_name=run_name,
                                            seed=seed,
                                            ema=option) if model_name != 'vit' else vit_command.format(i=counter,
                                                                                                       model=model_name,
                                                                                                       dataset=dataset,
                                                                                                       model_name=model_name,
                                                                                                       seed=seed)
            if option == 'ssm_2d':
                command_to_run += " --n_ssm=8 --ndim=16"
            print(command_to_run)
            counter += 1
            if counter > 7:
                counter = 0
