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

# Fully connected Model
def FC(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    input_size = len(data[0])
    hidden_layer_sizes = [150,100,50,1]
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_prob)
    return auc_score
# XGB Model
def XGB(data, labels, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=42)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred)
    return auc_score

# Logistic Regression Model
def LR(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    return auc

# Instances mean
def stupid_bag_embed(data):
    embedded_data = []
    for bag in data:
        bag = np.array(bag)
        embedded_data.append(np.mean(bag,axis=0))
    return embedded_data


a = 1
num_features = [int(x.round()) for x in np.linspace(1, 200, 10)]
C1_auc = []
C1_std = []
C2_auc = []
C2_std = []
FC_auc = []
FC_std = []
XGB_auc = []
XGB_std = []
LR_auc = []
LR_std = []
font = fm.FontProperties(family='Times New Roman', size=14)

# Create a figure and an Axes object
fig, ax = plt.subplots()

for nf in tqdm(num_features):
    current_auc_C1 = []
    current_auc_C2 = []
    current_auc_FC = []
    current_auc_XGB = []
    current_auc_LR = []
    for _ in range(15):
        datasetA, datasetB = generate_data(nf, a)
        labels = [1 for b in datasetA] + [0 for b in datasetB]

        current_auc_C1.append(C1(datasetA+datasetB, labels))
        current_auc_C2.append(C2(datasetA+datasetB, labels))

        data = stupid_bag_embed(datasetA+datasetB)

        current_auc_FC.append(FC(data, labels))
        current_auc_XGB.append(XGB(data, labels))
        current_auc_LR.append(LR(data, labels))


    C1_auc.append(np.mean(current_auc_C1))
    C1_std.append(np.std(current_auc_C1))
    C2_auc.append(np.mean(current_auc_C2))
    C2_std.append(np.std(current_auc_C2))
    FC_auc.append(np.mean(current_auc_FC))
    FC_std.append(np.std(current_auc_FC))
    XGB_auc.append(np.mean(current_auc_XGB))
    XGB_std.append(np.std(current_auc_XGB))
    LR_auc.append(np.mean(current_auc_LR))
    LR_std.append(np.std(current_auc_LR))


ax.plot(num_features,FC_auc, label='Fully Connected', color='#4ABD42')
ax.plot(num_features,XGB_auc, label='XGB', color='purple')
ax.plot(num_features,LR_auc, label='Logistic Regression', color='#FF6000')
ax.plot(num_features,C1_auc, label='C1', color='blue')
ax.plot(num_features,C2_auc, label='C2', color='red')
ax.set_xlim([10, 200])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('NF', fontproperties=font)
ax.set_ylabel('AUC', fontproperties=font)
ax.legend(loc='lower right', prop=font)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("AUC_vs_feat_num_plot.pdf", format="pdf")
plt.show()