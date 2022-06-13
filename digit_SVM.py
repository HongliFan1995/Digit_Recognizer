import numpy as np

from utils.utils import data_loading, write_predict
from utils.model import SVM_model

# load data
train, test, label = data_loading('data/train.csv', 'data/test.csv')      

# converting list to array
X_train = np.asarray(train)/255.0
y_train = np.asarray(label)

X_test = np.asarray(test)/255.0


#SVC for classification 
# define model and fit
linear = SVM_model('linear', C=1).fit(X_train, y_train)
rbf = SVM_model('rbf', C=1, gamma=1).fit(X_train, y_train)
poly = SVM_model('poly', C=1, degree=3).fit(X_train, y_train)
sig = SVM_model('sigmoid', C=1).fit(X_train, y_train)

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
write_predict("results/SVM/test_result_lin.csv", linear_pred)
write_predict("results/SVM/test_result_poly.csv", poly_pred)
write_predict("results/SVM/test_result_rbf.csv", rbf_pred)
write_predict("results/SVM/test_result_sig.csv", sig_pred)