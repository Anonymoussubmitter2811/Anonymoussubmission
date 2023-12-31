import numpy as np


NUM_OF_BAGS = 100
POISSON_LAMBDA = 3

def generate_number_poisson(lam):
    return np.random.poisson(lam)


def generate_binary_bernoulli(p):
    return np.random.choice([0, 1], p=[1 - p, p])


def generate_prob_vector(num_features, alpha, similarity_percent, prob_vector_A):
    prob_vector = []
    if not similarity_percent:
        for i in range(num_features):
            prob_vector.append(np.random.uniform(0, alpha))
    else:
        num_similar = int(num_features * similarity_percent / 100)
        similar_indices = np.random.choice(num_features, num_similar, replace=False)

        for i in range(num_features):
            if i in similar_indices:
                prob_vector.append(prob_vector_A[i])
            else:
                prob_vector.append(np.random.uniform(0, alpha))
    return prob_vector


def generate_instance(num_features, prob_vector):
    instance = []
    for i in range(num_features):
        instance.append(generate_binary_bernoulli(prob_vector[i]))
    return instance


def generate_bag(prob_vector,num_instances):
    bag = []
    for i in range(num_instances):
        bag.append(generate_instance(len(prob_vector), prob_vector))
    return bag


def generate_dataset(num_bags, prob_vector):
    data = []
    for i in range(num_bags):
        num_instances = generate_number_poisson(POISSON_LAMBDA)
        if num_instances != 0:
            data.append(generate_bag(prob_vector,num_instances))
    return data


def generate_data(NUM_OF_FEATURES, ALPHA):
    prob_vector_A = generate_prob_vector(NUM_OF_FEATURES, ALPHA, None, None)
    prob_vector_B = generate_prob_vector(NUM_OF_FEATURES, ALPHA, 80, prob_vector_A)

    return generate_dataset(NUM_OF_BAGS, prob_vector_A), generate_dataset(NUM_OF_BAGS, prob_vector_B)
