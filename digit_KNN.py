import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from utils.utils import data_loading, write_predict
from utils.model import KNN_model

# load data
train, test, label = data_loading('data/train.csv', 'data/test.csv')

#split train data into train and validation
X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=0.20)

#scale
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test)

#training and predictions
n_neighbors = 11
classifier = KNN_model(n_neighbors)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

#evaluate
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))

# write test results into csv file
Y_pred_test = classifier.predict(X_test)
write_predict(
    "results/KNN//test_result_KNN{}.csv".format(n_neighbors),
    Y_pred_test,
)