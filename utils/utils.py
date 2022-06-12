import csv
import numpy as np
import os

def data_loading(train_file, test_file):
    """
    Data loading function
    """
    train = []
    label = []
    with open(train_file, "r") as f:
        trainreader = csv.reader(f, delimiter = ",")
        
        for i, row in enumerate(trainreader):
            if i > 0:
                row = list(map(int, row))
                train.append(row[1:])
                label.append(row[0])
    
    test = []
    with open(test_file, "r") as f:
        testreader = csv.reader(f, delimiter = ",")

        for i, row in enumerate(testreader):
            if i > 0:
                row = list(map(int, row))
                test.append(row)

    return train, test, label

def write_predict(filename, results):
    """
    Write predictions
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ImageID", "Label"])
        for i, x in enumerate(results):
            writer.writerow([i+1, x])
    