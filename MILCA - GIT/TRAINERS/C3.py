import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from UTILS.utils_for_c_tests import clean_animal, clean_musk, bag_to_onehot
from UTILS.utils_for_c_tests import top_k_for_C3 as top_k
from CONFIGS.config import DATASETS_LOCATION, DATASETS_PARAMS
from TRAINERS.MainTrainer import train_eval

# This function handle the run of C3
def C3(dataset_name, lr, wd, drop_out):
    # Get data
    location = DATASETS_LOCATION[dataset_name]
    if dataset_name == "tiger" or dataset_name == "elephant" or dataset_name == "fox":
        data, labels = clean_animal(location)
    elif 'musk' in dataset_name:
        data, labels = clean_musk(location)

    # Train test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              stratify=labels)
    # Get significant features
    top_k_index = top_k([x for i, x in enumerate(train_samples) if train_labels[i] == 1],
                        [x for i, x in enumerate(train_samples) if
                         train_labels[i] == (0 if 'musk' in dataset_name else -1)])

    # Embed bags
    for i, bag in enumerate(train_samples):
        train_samples[i] = bag_to_onehot(bag, top_k_index)
    for i, bag in enumerate(test_samples):
        test_samples[i] = bag_to_onehot(bag, top_k_index)

    # train
    return train_eval(train_samples, train_labels, test_samples, test_labels, dataset_name, lr, wd,drop_out)


# Initiate Runs:
datasets_for_run = ['musk1', 'musk2', 'fox', 'tiger', 'elephant']
n_runs = 15
for dataset_name in datasets_for_run:
    lr = DATASETS_PARAMS[dataset_name]['lr']
    wd = DATASETS_PARAMS[dataset_name]['wd']
    drop_out = DATASETS_PARAMS[dataset_name]['drop_out']
    acc_train, acc_val, acc_test, auc_test = [], [], [], []
    for _ in tqdm(range(n_runs)):
        train_accuracy, val_accuracy, test_accuracy, test_auc = C3(dataset_name, lr, wd, drop_out)

        acc_train.append(train_accuracy)
        acc_val.append(val_accuracy)
        acc_test.append(test_accuracy)
        auc_test.append(test_auc)

    print()
    print()
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"FINAL RESULTS FOR {dataset_name.upper()}:")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"TRAIN:\t{np.mean(acc_train).round(3)} +- {np.std(acc_train).round(3) / (len(acc_train) ** .5)}")
    print(f"VAL:\t{np.mean(acc_val).round(3)} +- {np.std(acc_val).round(3) / (len(acc_val) ** .5)}")
    print(f"TEST:\t{np.mean(acc_test).round(3)} +- {np.std(acc_test).round(3) / (len(acc_test) ** .5)}")
    print()
    print()


