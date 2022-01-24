import pandas as pd
import random

import time

from labelers import knn

DATA_FILE = '../Data/credit_cards/creditcard.csv'

def read_data(data_file):
    data = pd.read_csv(data_file)
    return data

def split_data(data, ratios):
    different_classes = data.Class.unique()
    subsets_idcs = [[] for _ in ratios]
    
    for class_ in different_classes:
        indices_where_class = data.loc[data['Class'] == class_].index.to_list()
        random.shuffle(indices_where_class)
        n_in_class = len(indices_where_class)

        done_ratio = 0

        for i, ratio in enumerate(ratios):
            start_idx = round(done_ratio * n_in_class)
            end_idx = round((done_ratio + ratio) * n_in_class)
            done_ratio += ratio
            
            subsets_idcs[i] += indices_where_class[start_idx:end_idx]

    subsets = [data.iloc[subset_idcs].reset_index() for subset_idcs in subsets_idcs]

    return subsets

def prepare_data(data):
    # new_data = (input_vector, target)
    new_data = [None for _ in range(len(data))]

    for idx, row in data.iterrows():
        input_data = [row['V{}'.format(i + 1)] for i in range(28)]
        target = row['Class']
        new_data[idx] = (input_data, target)

    return new_data

def prepare_data(data):
    # new_data = (input_vector, target)
    new_data = [None for _ in range(len(data))]

    for idx, row in data.iterrows():
        input_data = [row['V{}'.format(i + 1)] for i in range(28)]
        target = row['Class']
        new_data[idx] = (input_data, target)

    return new_data

def main(data_file):
    start_time = time.time()
    data = read_data(data_file)
    print('read {}'.format(time.time() - start_time))
    train, test = split_data(data, (0.8, 0.2))
    print(train.size, test.size)
    
    start_time = time.time()
    train_lab, train_unlab = split_data(train, (0.3, 0.7))
    print('split {}'.format(time.time() - start_time))
    
    print(train_lab.size, train_unlab.size)
    
    start_time = time.time()
    
    test, train_lab, train_unlab = [prepare_data(d) for d in [test, train_lab, train_unlab]]

    print('prepare {}'.format(time.time() - start_time))
    
    
    # model = knn()
    # training_performance = model.train(train_lab)
    # test_performance = model.test(test)

    # model = knn()

    start_time = time.time()
    train_unlab = knn(train_lab, train_unlab, {'k': 5})
    print('knn {}'.format(time.time() - start_time))



if __name__ == '__main__':
    main(DATA_FILE)