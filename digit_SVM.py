#Importing the necessary packages and libaries
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
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

     #print(label[0:4])   
     #label_temp  = np.zeros(shape=(len(label), 10))
     #for idx, x in enumerate(label):
         #label_temp[idx][x] = 1


test = []
with open('./data/test.csv', newline='') as csvfile:
     testreader = csv.reader(csvfile, delimiter=',')     
     next(testreader)

     for row in testreader:
         row = list(map(int, row))
         test.append(row)        

# converting list to array
X_train = np.asarray(train)/255.0
y_train = np.asarray(label)

X_test = np.asarray(test)/255.0


#SVC for classification 
# define model and fit
linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

# retrieve the accuracy and print it for all 4 kernel functions
accuracy_lin = linear.score(X_train, y_train)
accuracy_poly = poly.score(X_train, y_train)
accuracy_rbf = rbf.score(X_train, y_train)
accuracy_sig = sig.score(X_train, y_train)
print("Accuracy Linear Kernel:", accuracy_lin)
print("Accuracy Polynomial Kernel:", accuracy_poly)
print("Accuracy Radial Basis Kernel:", accuracy_rbf)
print("Accuracy Sigmoid Kernel:", accuracy_sig)

#predict
linear_pred = linear.predict(X_test)
poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)
sig_pred = sig.predict(X_test)

# write test results into csv file
file = open("test_result_lin.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(linear_pred):
    writer.writerow([idx+1, x])
file.close()

file = open("test_result_poly.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(poly_pred):
    writer.writerow([idx+1, x])
file.close()

file = open("test_result_rbf.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(rbf_pred):
    writer.writerow([idx+1, x])
file.close()

file = open("test_result_sig.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(sig_pred):
    writer.writerow([idx+1, x])
file.close()