import numpy as np
from sklearn.model_selection import train_test_split

from utils.utils import data_loading, write_predict
from utils.model import CNN_model

# load data
train, test, label = data_loading('data/train.csv', 'data/test.csv')     

# converting list to array
train_arr = np.asarray(train)/255.0
label_arr = np.asarray(label)
train_arr = np.reshape(train_arr,(-1,28,28))

test_arr = np.asarray(test)/255.0
test_arr = np.reshape(test_arr,(-1,28,28))

X_train, X_val, y_train, y_val = train_test_split(
    train_arr, 
    label_arr, 
    test_size=0.05,
)

# build model
model = CNN_model()
model.summary()

hist = model.fit(
    X_train, 
    y_train, 
    epochs=10,
    validation_data = (X_val, y_val)    
)

#save model
model.save('models/digit_recognizer.h5')

# write test results into csv file
test_result = model(test_arr)
test_result = test_result.numpy()
test_result = np.argmax(test_result, axis=1)
write_predict("results/CNN/test_result.csv", test_result)