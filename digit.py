import tensorflow as tf
import csv
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

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
train_arr = np.asarray(train)/255.0
#print(train _arr[0][28*14:28*15])
label_arr = np.asarray(label)
#print(label_arr[0:4])
train_arr = np.reshape(train_arr,(-1,28,28))
#print(train _arr[0][14])

test_arr = np.asarray(test)/255.0
test_arr = np.reshape(test_arr,(-1,28,28))
#print(test_arr[0][14])

# build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# compile and train model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_arr, label_arr, epochs=10)

#save model
model.save('digit_recognizer.h5')

# evaluate the model 
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])


test_result = model.predict(test_arr)

# write test results into csv file
file = open("test_result.csv", "w")
writer = csv.writer(file)
writer.writerow(['ImageID','Label'])
for idx, x in enumerate(test_result):
    writer.writerow([idx+1, np.argmax(x)])
file.close()