import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import xgboost as xgb


from TOY.data_creator import generate_data
from TRAINERS.C1 import C1_toy as C1
from TRAINERS.C2 import C2_toy as C2


def FC(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    input_size = len(data[0])
    hidden_layer_sizes = [150,100,50,1]
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    return auc_score

def XGB(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score

def LR(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc

def stupid_bag_embed(data):
    embedded_data = []
    for bag in data:
        bag = np.array(bag)
        embedded_data.append(np.mean(bag,axis=0))
    return embedded_data


alpha = np.logspace(np.log10(10**-2), np.log10(1), 10)
num_features = 150
C1_auc = []
C2_auc = []
FC_auc = []
XGB_auc = []
LR_auc = []

for a in tqdm(alpha):
    current_auc_C1 = []
    current_auc_C2 = []
    current_auc_FC = []
    current_auc_XGB = []
    current_auc_LR = []
    for _ in range(10):
        datasetA, datasetB = generate_data(num_features, a)
        labels = [1 for b in datasetA] + [0 for b in datasetB]

        current_auc_C1.append(C1(datasetA+datasetB, labels))
        # current_auc_C1.append(0.8)
        current_auc_C2.append(C2(datasetA+datasetB, labels))
        # current_auc_C2.append(0.7)

        data = stupid_bag_embed(datasetA+datasetB)

        current_auc_FC.append(FC(data, labels))
        # current_auc_FC.append(0.1)
        current_auc_XGB.append(XGB(data, labels))
        # current_auc_XGB.append(0.2)
        current_auc_LR.append(LR(data, labels))
        # current_auc_LR.append(0.3)


    C1_auc.append(np.mean(current_auc_C1))
    C2_auc.append(np.mean(current_auc_C2))
    FC_auc.append(np.mean(current_auc_FC))
    XGB_auc.append(np.mean(current_auc_XGB))
    LR_auc.append(np.mean(current_auc_LR))

font = fm.FontProperties(family='Times New Roman', size=14)

# Create a figure and an Axes object
fig, ax = plt.subplots()
ax.plot(alpha,FC_auc, label='Fully Connected', color='#4ABD42')
ax.plot(alpha,XGB_auc, label='XGB', color='purple')
ax.plot(alpha,LR_auc, label='Logistic Regression', color='#FF6000')
ax.plot(alpha,C1_auc, label='C1', color='blue')
ax.plot(alpha,C2_auc, label='C2', color='red')


ax.plot([0, 200], [0.5, 0.5], 'k--', label='Random')
ax.set_xscale('log')
ax.set_xlim([10**-2, 1])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel(r'$\alpha$', fontproperties=font)  # Use LaTeX formatting for alpha
ax.set_ylabel('AUC', fontproperties=font)
ax.legend(loc='lower right', prop=font)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("AUC_vs_Alpha_plot.pdf", format="pdf")
plt.show()
