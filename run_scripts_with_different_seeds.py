datasets = ["CIFAR100","T-IMNET"]
# dataset = "T-IMNET"
command = "CUDA_VISIBLE_DEVICES={i} python main.py --model {model} --dataset {dataset} " \
          "--project {dataset}_s4nd_comparison --name {ema},seed={seed} --seed {seed} --ema {ema}" \
          " --smooth_v_as_well --use_relative_pos_embedding" \
          " --s4nd_config \"/media/data2/ethan_baron/ssm_2d_patches/s4nd_configs/s4nd_current.yaml\" " \
          "--n_ssm {n_ssm} --ndim {ndim}"
models = {'vit': ['ssm_2d', 's4nd']}
seed_amount = 3
counter = 0
for dataset in datasets:
    for model_name, ema_options in models.items():
        for option in ema_options:
            for seed in range(seed_amount):
                    command_to_run = command.format(i=counter, model=model_name, dataset=dataset, seed=seed, ema=option,
                                                    n_ssm=2, ndim=8)
                    print(command_to_run)
                    counter += 1
                    if counter > 7:
                        counter = 0
