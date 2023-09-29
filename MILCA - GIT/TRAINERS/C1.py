import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from UTILS.utils_for_c_tests import clean_animal, clean_musk, get_dist
from UTILS.utils_for_c_tests import top_k_for_C1 as top_k
from CONFIGS.config import DATASETS_LOCATION


# This function return a bag score (mean on instances)
def get_score_count_only(bag, best_feat):
    inst_count, feat_count = len(bag), 0
    for instance in bag:
        for i in best_feat:
            feat_count += instance[i]
    score = feat_count / inst_count
    return score


# This function handle the run of C1 for the toy test
def C1_toy(data, labels):
    # Train test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              stratify=labels, random_state=42)
    # Get significant features
    top_k_index = top_k([x for i, x in enumerate(train_samples) if train_labels[i] == 1],
                        [x for i, x in enumerate(train_samples) if train_labels[i] == 0])

    # get test scores
    test_predicted_scores = []
    for bag in test_samples:
        test_predicted_scores.append(get_score_count_only(bag, top_k_index))

    auc_score = roc_auc_score(test_labels, test_predicted_scores)
    return auc_score


def C1(dataset_name, rs):
    # Get data
    location = DATASETS_LOCATION[dataset_name]
    if dataset_name == "tiger" or dataset_name == "elephant" or dataset_name == "fox":
        data, labels = clean_animal(location)
    elif 'musk' in dataset_name:
        data, labels = clean_musk(location)

    # Train test split
    train_samples, test_samples, train_labels, test_labels = train_test_split(data, labels, test_size=0.2,
                                                                              stratify=labels, random_state=rs)
    # Get significant features
    top_k_index = top_k([x for i, x in enumerate(train_samples) if train_labels[i] == 1],
                        [x for i, x in enumerate(train_samples) if
                         train_labels[i] == (0 if 'musk' in dataset_name else -1)])

    test_predicted_scores = []
    train_predicted_scores = []

    for bag in test_samples:
        test_predicted_scores.append(get_score_count_only(bag, top_k_index))
    for bag in train_samples:
        train_predicted_scores.append(get_score_count_only(bag, top_k_index))

    # Calculate accuracy for different thresholds
    sorted_predicted_scores_train = sorted(train_predicted_scores) + [max(train_predicted_scores) + 1]
    accus_train = []
    accus_test = []
    thresholds = []
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

    return max(accus_test)


# Initiate Runs:
datasets = ['musk1', 'musk2', 'fox', 'tiger', 'elephant']

for dataset_name in datasets:
    test_accs = []
    for rs in tqdm(range(95, 105)):
        acc = C1(dataset_name, rs)
        test_accs.append(acc)

    print(dataset_name.upper())
    print("Mean Accuracy:", f"{np.mean(test_accs).round(3)}±{np.std(test_accs).round(3)}")
