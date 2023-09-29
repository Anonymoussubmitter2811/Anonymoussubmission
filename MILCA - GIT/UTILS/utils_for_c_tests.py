import numpy as np
import scipy.stats as stats

# This function gets the raw MUSK data and clens it into Data[Bag[Instance[Feature]]] structure.
def clean_musk(csv_file):
    data = []
    labels = []

    with open(csv_file) as f:
        last_bag = ''
        lines = f.readlines()

        for line in lines:
            bag_name, _, *instance_vector, label = line.strip()[:-1].split(",")
            instance_vector = [int(value) for value in instance_vector]

            if bag_name == last_bag:
                data[-1].append(instance_vector)
            else:
                data.append([instance_vector])
                labels.append(int(label))

            last_bag = bag_name

    return data, labels

# This function gets the raw FOX,TIGER,ELEPHANT data and clens it into Data[Bag[Instance[Feature]]] structure.
def clean_animal(csv_file):
    data = []
    labels = []

    with open(csv_file) as f:
        last_bag = ''
        lines = f.readlines()[1:]

        for line in lines:
            instance_info, *instance_vector = line.strip().split(" ")
            _, bag_name, label = instance_info.split(":")
            instance_vector = [float(value.split(":")[1]) for value in instance_vector]

            if bag_name == last_bag:
                data[-1].append(instance_vector)
            else:
                data.append([instance_vector])
                labels.append(int(label))

            last_bag = bag_name

    return data, labels

# This function returns the significant features under the C1 restrictions.
def top_k_for_C1(datasetA, datasetB):
    best_feat = []
    for i in range(len(datasetA[0][0])):
        dist_i_A, dist_i_B = get_dist(datasetA, i), get_dist(datasetB, i)

        statistic, p_value = stats.mannwhitneyu(dist_i_A, dist_i_B)
        if p_value < 0.05 and (np.mean(dist_i_A) > np.mean(dist_i_B)):
            best_feat.append(i)
    return best_feat

# This function returns the significant features under the C2 restrictions.
def top_k_for_C2(datasetA, datasetB):
    best_feat = {}
    for i in range(len(datasetA[0][0])):
        dist_i_A, dist_i_B = get_dist(datasetA, i), get_dist(datasetB, i)
        statistic, p_value = stats.mannwhitneyu(dist_i_A, dist_i_B)
        if p_value < 0.05:
            if (np.mean(dist_i_A) > np.mean(dist_i_B)):
                best_feat[i] = 1
            else:
                best_feat[i] = -1
    return best_feat

# This function returns the significant features under the C3 restrictions.
def top_k_for_C3(datasetA, datasetB):
    best_feat = []
    for i in range(len(datasetA[0][0])):
        dist_i_A, dist_i_B = get_dist(datasetA, i), get_dist(datasetB, i)

        statistic, p_value = stats.mannwhitneyu(dist_i_A, dist_i_B)

        if p_value < 0.05:
            best_feat.append(i)
    return best_feat

# This function gets a dataset and an index, and returns the distribution(list of means) of the index in the bags.
def get_dist(dataset, index):
    dist = []
    for bag in dataset:
        feat_count = 0
        for instance in bag:
            feat_count += instance[index]
        dist.append(feat_count / len(bag))
    return dist

# This function gets a bag and a list of features and returns a vector of means.
def bag_to_onehot(bag, top_k):
    num_instances = len(bag)
    num_words = len(top_k)
    onehot = [0] * num_words

    for instance in bag:
        for j, index in enumerate(top_k):
            onehot[j] += instance[index]

    return [x/num_instances for x in onehot]
