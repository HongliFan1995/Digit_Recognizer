import numpy as np
import csv

# load data 
train = []
label = []
with open('./data/train.csv', newline='') as csvfile:
     trainreader = csv.reader(csvfile, delimiter=',')     
     next(trainreader)

     for row in trainreader:
         row = list(map(int, row))
         train.append(row[1:])
         label.append(row[0])

test = []
with open('./data/test.csv', newline='') as csvfile:
     testreader = csv.reader(csvfile, delimiter=',')     
     next(testreader)

     for row in testreader:
         row = list(map(int, row))
         test.append(row)      



#split train data into train and validation
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train, label, test_size=0.20)

#scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(test)

#training and predictions
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=17)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_val)

#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_val, y_pred))
print(classification_report(y_val, y_pred))


# write test results into csv file
Y_pred_test = classifier.predict(X_test)
file = open("./results/test_result_KNN17.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(Y_pred_test):
    writer.writerow([idx+1, x])
file.close()

