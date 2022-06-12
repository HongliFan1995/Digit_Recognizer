import tensorflow as tf
import csv
import numpy as np
from tensorflow.keras import datasets, layers, models

model = tf.keras.models.load_model('digit_recognizer.h5')

test = []
with open('./data/test.csv', newline='') as csvfile:
     testreader = csv.reader(csvfile, delimiter=',')     
     next(testreader)

     for row in testreader:
         row = list(map(int, row))
         test.append(row)        

     

# converting list to array
test_arr = np.asarray(test)/255.0
test_arr = np.reshape(test_arr,(-1,28,28))

test_result = model.predict(test_arr)

# write test results into csv file
file = open("test_result.csv", "w")
writer = csv.writer(file)
for idx, x in enumerate(test_result):
    writer.writerow([idx+1, np.argmax(x)])
file.close()