import itertools

# datasets = ['CIFAR100']  # ["T-IMNET"]
# dataset = "T-IMNET"
command = "python main.py --model {model} --dataset {dataset} " \
          "--project {dataset}_Cait_Comparison --name {name} --seed {seed} --ema {ema} " \
          "--n_ssm {n_ssm} --ndim {ndim} --sd {sd} --gpu {gpu}"
# models = {'vit': ['ssm_2d', 's4nd']}
params = {'sd': [0.1, 0.15, 0.2, 0.3], 'n_ssm': [2, 8], 'dataset': [''], 'model': ['convit'],
          'ema': ['ssm_2d'], 'seed': [0], 'ndim': [16]}

seed_amount = 1
counter = 0
# Go through all options in params, meaning for each dictionary in params there is a list,
# cross all the options between the lists
for option in itertools.product(*params.values()):
    # now zip the keys and the options together
    keys_and_options = dict(zip(params.keys(), option))
    keys_and_options['gpu'] = counter

    # convert dictionary to readable string
    name = "convit_ssm_2d_"
    for key, value in keys_and_options.items():
        if key!='gpu' and len(params[key]) > 1 :
            name += str(key) + "=" + str(value) + "_"
    name = name[:-1]
    keys_and_options['name'] = name

    print(command.format(**keys_and_options))
    counter += 1
    if counter > 7:
        counter = 0

