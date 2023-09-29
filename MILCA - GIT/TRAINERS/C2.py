import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from CONFIGS.config import DATASETS_LOCATION
from UTILS.utils_for_c_tests import clean_animal, clean_musk
from UTILS.utils_for_c_tests import top_k_for_C2 as top_k


# This function return a bag score (binary weighted mean on instances)
def get_score_count(bag, best_feat, theta):
    inst_count, feat_count = len(bag), 0
    for instance in bag:
        for i in range(len(instance)):
            if i in best_feat.keys():
                if best_feat[i] == 1:
                    feat_count += instance[i]
                else:
                    feat_count -= theta * instance[i]
    return feat_count / inst_count


# This function handle the run of C2 for the toy test
def C2_toy(data, labels):
    # Train test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              random_state=42, stratify=labels)
    top_k_index = top_k([x for i, x in enumerate(train_samples) if train_labels[i] == 1],
                        [x for i, x in enumerate(train_samples) if
                         train_labels[i] == 0])

    train_aucs = []
    test_aucs = []
    betas = [x / 100 for x in range(0, 1001, 5)]
    for beta in betas:
        train_predicted_scores = []
        test_predicted_scores = []
        for bag in train_samples:
            train_predicted_scores.append(get_score_count(bag, top_k_index, beta))
        for bag in test_samples:
            test_predicted_scores.append(get_score_count(bag, top_k_index, beta))

        train_aucs.append(roc_auc_score(train_labels, train_predicted_scores))
        test_aucs.append(roc_auc_score(test_labels, test_predicted_scores))
    auc_on_best_beta = test_aucs[np.argmax(train_aucs)]

    return auc_on_best_beta


# This function handle the run of C2
def C2(dataset_name, rs):
    # Get data
    location = DATASETS_LOCATION[dataset_name]
    if dataset_name == "tiger" or dataset_name == "elephant" or dataset_name == "fox":
        data, labels = clean_animal(location)
    elif 'musk' in dataset_name:
        data, labels = clean_musk(location)

    # Train test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              random_state=rs, stratify=labels)
    # Get significant features
    top_k_index = top_k([x for i, x in enumerate(train_samples) if train_labels[i] == 1],
                        [x for i, x in enumerate(train_samples) if
                         train_labels[i] == (0 if 'musk' in dataset_name else -1)])
    # Find best Beta
    train_aucs = []
    test_aucs = []
    betas = [x / 100 for x in range(0, 1001, 5)]
    for beta in betas:
        train_predicted_scores = []
        test_predicted_scores = []
        for bag in train_samples:
            train_predicted_scores.append(get_score_count(bag, top_k_index, beta))
        for bag in test_samples:
            test_predicted_scores.append(get_score_count(bag, top_k_index, beta))

        train_aucs.append(roc_auc_score(train_labels, train_predicted_scores))
        test_aucs.append(roc_auc_score(test_labels, test_predicted_scores))
    best_beta = betas[np.argmax(train_aucs)]
    auc_on_best_beta = test_aucs[np.argmax(train_aucs)]

    test_predicted_scores = []
    for bag in test_samples:
        test_predicted_scores.append(get_score_count(bag, top_k_index, best_beta))

    sorted_predicted_scores_train = sorted(train_predicted_scores) + [max(train_predicted_scores) + 1]
    accus_train = []
    accus_test = []
    thresholds = []

    # Calculate accuracy for different thresholds
    for i in range(len(sorted_predicted_scores_train) - 1):
        threshold = (sorted_predicted_scores_train[i] + sorted_predicted_scores_train[i + 1]) / 2
        predicted_labels_train = [1 if score > threshold else (0 if 'musk' in dataset_name else -1) for score in
                                  train_predicted_scores]
        predicted_labels_test = [1 if score > threshold else (0 if 'musk' in dataset_name else -1) for score in
                                 test_predicted_scores]
        accuracy_train = accuracy_score(train_labels, predicted_labels_train)
        accuracy_test = accuracy_score(test_labels, predicted_labels_test)
        accus_train.append(accuracy_train)
        accus_test.append(accuracy_test)

        thresholds.append(threshold)

    return max(accus_test), best_beta, auc_on_best_beta


# Initiate Runs:
datasets_for_run = ['musk1', 'musk2', 'fox', 'tiger', 'elephant']

for dataset_name in datasets_for_run:
    test_accs = []
    dif_betas = []
    dif_aucs = []

    for rs in tqdm(range(95, 105)):
        acc = C2(dataset_name, rs)
        test_accs.append(acc[0])
        dif_betas.append(acc[1])
        dif_aucs.append(acc[2])

    print(dataset_name.upper())
    print("Mean Accuracy:", f"{np.mean(test_accs).round(3)}±{np.std(test_accs).round(3)}")
