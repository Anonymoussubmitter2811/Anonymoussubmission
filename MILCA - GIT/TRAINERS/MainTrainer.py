import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from CONFIGS.config import DATASETS_PARAMS
from TRAINERS.CountingAttentionModel import SigmoidWeightLogisticRegression

#This function train the Model for C3.
def train_eval(vectors_data_train, labels_data_train, vectors_data_test, labels_data_test, dataset_name, lr, wd, drop_out):
    validation_split = DATASETS_PARAMS[dataset_name]['validation_rate']
    num_epochs = DATASETS_PARAMS[dataset_name]['num_epochs']

    # Convert to tensors
    vectors_data_train = torch.tensor(vectors_data_train, dtype=torch.float32)
    labels_data_train = torch.tensor(labels_data_train, dtype=torch.float32)
    vectors_data_test = torch.tensor(vectors_data_test, dtype=torch.float32)
    labels_data_test = torch.tensor(labels_data_test, dtype=torch.float32)

    # Train Val split
    vectors_data_train, vectors_data_val, labels_data_train, labels_data_val = train_test_split(
        vectors_data_train, labels_data_train,
        test_size=validation_split,
        stratify=labels_data_train
    )

    # Fix labels
    if sorted(np.unique(labels_data_train)) == [-1, 1]:
        labels_data_train = ((labels_data_train + 1) / 2).long()
        labels_data_val = ((labels_data_val + 1) / 2).long()
        labels_data_test = ((labels_data_test + 1) / 2).long()

    # Initiate Model
    input_size = vectors_data_train.shape[1]
    model = SigmoidWeightLogisticRegression(input_size, drop_out)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_losses = []
    val_losses = []
    train_accs, val_accs = [], []

    n_epoch_wo_imp, best_val_acc = 0, -1
    es_patience = 400

    # Train Model
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(vectors_data_train)
        loss = criterion(outputs, labels_data_train.unsqueeze(1).float())
        outputs = torch.sigmoid(outputs)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_acc = (outputs.round() == labels_data_train.unsqueeze(1)).float().mean().item()
        train_accs.append(train_acc)

        # Validation
        with torch.no_grad():
            val_outputs = model(vectors_data_val)
            val_loss = criterion(val_outputs, labels_data_val.unsqueeze(1).float())
            val_outputs = torch.sigmoid(val_outputs)
            val_losses.append(val_loss.item())
            val_acc = (val_outputs.round() == labels_data_val.unsqueeze(1)).float().mean().item()
            val_accs.append(val_acc)

        if val_acc > best_val_acc:
            n_epoch_wo_imp = 0
            best_val_acc = val_acc

        else:
            n_epoch_wo_imp += 1

        if n_epoch_wo_imp == es_patience:
            # print(f"ES triggered at epoch {epoch + 1}.")
            break

    model.eval()
    # Get predictions
    train_predictions = torch.sigmoid(model(vectors_data_train).squeeze()).detach().numpy()
    val_predictions = torch.sigmoid(model(vectors_data_val)).squeeze().detach().numpy()
    test_predictions = torch.sigmoid(model(vectors_data_test)).squeeze().detach().numpy()

    accs_test = []
    thresholds = []
    sorted_predicted_scores_test = sorted(test_predictions) + [max(test_predictions) + 1]
    # Get best Threshold
    for i in range(len(sorted_predicted_scores_test) - 1):
        threshold = (sorted_predicted_scores_test[i] + sorted_predicted_scores_test[i + 1]) / 2
        predicted_labels_test = [1 if score > threshold else 0 for score in test_predictions]
        accuracy_test = accuracy_score(labels_data_test, predicted_labels_test)
        accs_test.append(accuracy_test)
        thresholds.append(threshold)

    best_threshold = thresholds[np.argmax(accs_test)]

    threshold = best_threshold

    train_binary_predictions = [1 if score > threshold else 0 for score in train_predictions]
    train_roc_auc = roc_auc_score(labels_data_train.numpy(), train_predictions)
    train_accuracy = accuracy_score(labels_data_train.numpy(), train_binary_predictions)

    val_binary_predictions = [1 if score > threshold else 0 for score in val_predictions]
    val_roc_auc = roc_auc_score(labels_data_val.numpy(), val_predictions)
    val_accuracy = accuracy_score(labels_data_val.numpy(), val_binary_predictions)

    test_binary_predictions = [1 if score > threshold else 0 for score in test_predictions]
    test_roc_auc = roc_auc_score(labels_data_test.numpy(), test_predictions)
    test_accuracy = accuracy_score(labels_data_test.numpy(), test_binary_predictions)

    # Informative Prints
    # print()
    # print("~~~~~~~~~~~~~~~~~~~~~~~")
    #
    # print("Train:")
    # print(f"\tAccuracy: {train_accuracy}")
    # print(f"\tROC AUC: {train_roc_auc}")
    # print(
    #     f"\tRandom Val Accuracy: {(labels_data_train.unique(return_counts=True)[1].max() / labels_data_train.shape[0]).item()}")
    #
    # print("Val:")
    # print(f"\tAccuracy: {val_accuracy}")
    # print(f"\tROC AUC: {val_roc_auc}")
    # print(
    #     f"\tRandom Val Accuracy: {(labels_data_val.unique(return_counts=True)[1].max() / labels_data_val.shape[0]).item()}")
    #
    # print("Test:")
    # print(f"\tAccuracy: {test_accuracy}")
    # print(f"\tROC AUC: {test_roc_auc}")
    # print(
    #     f"\tRandom Val Accuracy: {(labels_data_test.unique(return_counts=True)[1].max() / labels_data_test.shape[0]).item()}")

    # Loss plot
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    return train_accuracy, val_accuracy, test_accuracy, test_roc_auc
