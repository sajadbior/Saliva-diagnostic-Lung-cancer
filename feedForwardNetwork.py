import pandas as pd
from sklearn.preprocessing import RobustScaler
import random
import keras
from keras import models
from keras import layers
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import numpy as np
import sys


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu', input_shape=(18,)))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))
    model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_dataset(gf, gt):

  Xg, ys = [], []
  lst = gf.shape[0] + gt.shape[0]
  f_counter = 0
  t_counter = 0

  for i in range(0, lst):
    rnd = random.randint(0, 100)
    if (rnd >= 50 and f_counter < gf.shape[0]) or (rnd < 50 and t_counter >= gt.shape[0]):
        z = gf.iloc[f_counter, :]
        Xg.append(z)
        ys.append(0)
        f_counter += 1

    elif(rnd < 50 and t_counter < gt.shape[0]) or (rnd >= 50 and f_counter >= gf.shape[0]):
        z = gt.iloc[t_counter, :]
        Xg.append(z)
        ys.append(1)
        t_counter += 1

  return np.array(Xg), np.array(ys)

case_0_df = pd.read_csv('data-new/case-0.csv')

case_1_df = pd.read_csv('data-new/case-1.csv')

case_0_df.drop(['C? Mean-mir 20a', "Metastatic", "Surgery", "Chemotherapy", "Radiotherapy", "sex", "Age", "Ethinity", "smoking ", "Alchoholic", "Adiction", "Cardia", "A.Rumathoid", "Diabete","Pulonary"], axis=1)
case_1_df.drop(['C? Mean-mir 20a', "Metastatic", "Surgery", "Chemotherapy", "Radiotherapy", "sex", "Age", "Ethinity", "smoking ", "Alchoholic", "Adiction", "Cardia", "A.Rumathoid", "Diabete","Pulonary"], axis=1)


g_scale_column = ['C? Mean-mir let 7a','C? Mean-mir 221', ]

g_scaler = RobustScaler()
g_scaler = g_scaler.fit(case_0_df[g_scale_column])
case_0_df.loc[:, g_scale_column] = g_scaler.transform(case_0_df[g_scale_column].to_numpy())
g_scaler=g_scaler.fit(case_1_df[g_scale_column])
case_1_df.loc[:, g_scale_column] = g_scaler.transform(case_1_df[g_scale_column].to_numpy())

Xg, y = create_dataset(case_0_df, case_1_df)
print(y)
thesh = 25
train_data = Xg[0:thesh, :]

train_lable = y[0:thesh]


num_epochs = 100
all_scores = []

model = build_model()
history = model.fit(train_data, train_lable,
                        epochs=num_epochs, batch_size=20, verbose=0)
history_dict = history.history
history_dict = history.history
acc = history.history['accuracy']
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(acc) + 1)
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(acc) + 1)

test_data = Xg[thesh:, :]
#test_data = Xg[32:, 1:]

test_lable = y[thesh:]

print(test_lable)
score = model.evaluate(test_data, test_lable, verbose=1)
print("Test Accuracy:", score[1])

preds = model.predict(test_data)

y_pred = np.where(preds < 0.5, 0, 1)

auc = roc_auc_score(test_lable, preds)
print('AUC: %.2f' % auc)
fpr1, tpr1, thresholds1 = roc_curve(test_lable, preds)
optimal_idx1 = np.argmax(tpr1 - fpr1)
optimal_idx2 = np.argmin(np.sqrt((np.power((1-tpr1), 2))+ (np.power((1-(1-fpr1)), 2))))
# print("Roc", thresholds1[optimal_idx1])
# print("Roc", thresholds1[optimal_idx2])
plt.plot(fpr1, tpr1, label='ROC curve (area = {0:0.2f})'''.format(auc), color='black')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
tn, fp, fn, tp = metrics.confusion_matrix(test_lable, y_pred).ravel()
confusion_matrix = metrics.confusion_matrix(test_lable, y_pred)
print(confusion_matrix)
Specificity = tn / (tn + fp)
Sensitivity = tp / (fn + tp)
Accuracy = (tp+tn)/ (tp+tn+fn+fp)
print("Specificity is ", Specificity)
print("Sensitivity is", Sensitivity)
print("Accuracy is", Accuracy)