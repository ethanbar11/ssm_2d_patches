datasets = ['CIFAR100']#["T-IMNET"]
# dataset = "T-IMNET"
command = "CUDA_VISIBLE_DEVICES={i} python main.py --model {model} --dataset {dataset} " \
          "--project {dataset}_Cait_Comparison --name {ema}_no_v_smoothing --seed {seed} --ema {ema}" \
          " --s4nd_config \"/media/data2/ethan_baron/ssm_2d_patches/s4nd_configs/s4nd_current.yaml\" " \
          "--n_ssm {n_ssm} --ndim {ndim}"
# models = {'vit': ['ssm_2d', 's4nd']}
models = {'cait': ['s4nd', 'ssm_2d','ema']}
seed_amount = 1
counter = 0
for dataset in datasets:
    for model_name, ema_options in models.items():
        for option in ema_options:
            for seed in range(seed_amount):
                command_to_run = command.format(i=counter, model=model_name, dataset=dataset, seed=seed, ema=option,
                                                n_ssm=8, ndim=32)
                print(command_to_run)
                counter += 1
                if counter > 7:
                    counter = 0
