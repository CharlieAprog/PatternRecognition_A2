def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, Birch, OPTICS
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.model_selection import KFold


def make_scatter_plot(data, labels, title, save_plot=False, dimension=2, cluster=[]):
    fig = plt.figure()
    if dimension == 3:

        ax = fig.add_subplot(projection='3d')
        ax.set_zlabel('PC3')

        if len(cluster) == 0:
            classes = list(set(np.array(labels.values)))
            for class_ in classes:
                ax.scatter(data[labels==class_][0], data[labels==class_][1], data[labels==class_][2], label=class_)
        else:
            ax.scatter(data[0], data[1], data[2], c=cluster)

    else:
        ax = fig.add_subplot()
        if len(cluster) == 0:
            classes = list(set(np.array(labels.values)))
            for class_ in classes:
                ax.scatter(data[labels==class_][0], data[labels==class_][1], label=class_)
        else:
            ax.scatter(data[0], data[1], c=cluster)

    plt.title(title)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    if save_plot:
        plt.savefig(f'../plots/numeric/{title}.jpg')
    else:
        plt.show()

def visualise_data(x, y, dims=2, save=False, title=''):
    pca = sklearnPCA(n_components=dims) #2-dimensional PCA
    pca_transformed = pd.DataFrame(pca.fit_transform(x))
    pca_transformed.index = y.index
    make_scatter_plot(pca_transformed, y,f'{title}_PCA', dimension=dims, save_plot=save)

    lda = LDA(n_components=dims) #2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(x, y))
    lda_transformed.index = y.index
    make_scatter_plot(lda_transformed, y,f'{title}_LDA',dimension=dims, save_plot=save)

def visualise_cluster(x, cluster_lables, dims=2, save=False, title='Clustering'):
    pca = sklearnPCA(n_components=dims)
    transformed = pd.DataFrame(pca.fit_transform(x))
    make_scatter_plot(transformed, [],f'{title}',dimension=dims, save_plot=save, cluster=cluster_lables)

def preprocess_data(model, x_train, x_test, y_train):
    x_train = model.fit_transform(x_train, y_train)
    x_test = model.transform(x_test)
    return x_train, x_test

def determine_best_col_num(model, classifier, x_train, y_train, max_n, plot=False):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, stratify=y_train, random_state=69)
    f1_scores = []
    num_cols = range(2, max_n)
    for n in tqdm(num_cols, leave=False):
        model.set_params(n_components=n)
        x_train_new, x_test_new = preprocess_data(model, x_train, x_test, y_train)
        pred = run_model(classifier, x_train_new, x_test_new, y_train)
        acc, f1 = get_score(pred, y_test)
        f1_scores.append(f1)

    index = np.argmax(np.array(f1_scores))
    print(f'smallest n:{num_cols[index]} with score of:{f1_scores[index]}')
    if plot:
        plt.figure()
        plt.plot(num_cols, f1_scores)
        plt.ylabel('f1 score')
        plt.xlabel('number of components')
        plt.title(plot)
        plt.savefig(f'../plots/numeric/{plot}.jpg')
        plt.show()
    return index + 2

def run_model(model, x_train, x_test, y_train):
    model.fit(x_train, y_train.to_numpy())
    return model.predict(x_test)

def classify_data(x_train, x_test, y_train, y_test):
    KNN = KNeighborsClassifier(n_neighbors=5)
    y_predKNN = run_model(KNN, x_train, x_test, y_train)
    knn_accuracy, knn_f1 = get_score(y_predKNN, y_test)

    tree = DecisionTreeClassifier(random_state=69)
    y_predTree = run_model(tree, x_train, x_test, y_train)
    tree_accuracy, tree_f1 = get_score(y_predTree, y_test)

    log = LogisticRegression(max_iter=400, solver='liblinear', random_state=69)
    y_predlog = run_model(log, x_train, x_test, y_train)
    log_accuracy, log_f1 = get_score(y_predlog, y_test)


    print(f'knn accuracy: {knn_accuracy}, knn f1: {knn_f1}')
    print(f'tree accuracy: {tree_accuracy}, tree f1: {tree_f1}')
    print(f'log accuracy:{log_accuracy}, log f1: {log_f1}')

def final_classify_data(x, y):
    KNN = KNeighborsClassifier(n_neighbors=5)
    scores = cross_validate(KNN, x, y, cv=10, return_train_score=True)
    knn_accuracy = np.mean(scores['test_score'])


    tree = DecisionTreeClassifier(criterion='gini', splitter='best', random_state=69)
    scores = cross_validate(tree, x, y, cv=10, return_train_score=True)
    tree_accuracy = np.mean(scores['test_score'])

    log = LogisticRegression(solver='liblinear', C = 1e-05, penalty = 'l2', random_state=69)
    scores = cross_validate(log, x, y, cv=10, return_train_score=True)
    log_accuracy = np.mean(scores['test_score'])


    print(f'knn accuracy: {knn_accuracy}')
    print(f'tree accuracy: {tree_accuracy}')
    print(f'log accuracy:{log_accuracy}')

def ensemble(x, y):
    KNN = KNeighborsClassifier(n_neighbors=5)
    tree = DecisionTreeClassifier(criterion='gini', splitter='best')
    log = LogisticRegression(penalty='l2', C = 1e-05, solver='liblinear', random_state=69)

    x = x.to_numpy()
    y = y.to_numpy()
    splits = 5
    kf = KFold(n_splits=splits)
    kf.get_n_splits(x)
    avg_accuracy = []
    
    total_accuracies = []
    for train_index, test_index in kf.split(x):
        ###print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #training
        KNN = KNN.fit(x_train, y_train)
        tree = tree.fit(x_train, y_train)
        log = log.fit(x_train, y_train)

        predictions = []
        testing_instances = 0
        ###manual ensemble###
        knn_pred = KNN.predict(x_test)
        tree_pred = tree.predict(x_test)
        log_pred = log.predict(x_test)

        accuracies = []
        for index in range(0, len(knn_pred)):
            knn_prediction = knn_pred[index]  
            tree_prediction = tree_pred[index] 
            log_prediction = log_pred[index]   
            ###majority vote###
            vote = 0
            if (knn_prediction ==  y_test[index]):
                vote += 1
            if(tree_prediction ==  y_test[index]):
                vote += 1
            if(log_prediction ==  y_test[index]):
                vote += 1
            if(vote >= 2):
                accuracies.append(1)
            else: 
                accuracies.append(0)
            ###print(vote, end = '')
        ####accuracy within kfold###
        ###print('')
        accuracy = sum(accuracies) / len(knn_pred)
        total_accuracies.append(accuracy)
        ###print(f'Accuracies within split: {accuracy}')
    ###overall accuracy###
    accuracy = sum(total_accuracies) / splits

    print(f'Ensemble: {accuracy}')

def get_score(pred, y_test):
    accuracy = round(accuracy_score(y_test, pred), 3)
    f1 = round(f1_score(y_test, pred, average="macro", zero_division=0), 3)
    return accuracy, f1

def calculate_kn_distance(x,k):

    kn_distance = []
    for i in range(len(x)):
        eucl_dist = []
        for j in range(len(x)):
            eucl_dist.append(
                math.sqrt(
                    ((x[i,0] - x[j,0]) ** 2) +
                    ((x[i,1] - x[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance


def cluster_data_search(x_train, x_test):
    complete_data = np.concatenate((x_train, x_test))
    #norm = (complete_data - np.min(complete_data))/np.ptp(complete_data)

    #print(np.max(norm), np.min(norm))
    # eps_dist = calculate_kn_distance(norm,20)
    # plt.hist(eps_dist,bins=30)
    # plt.ylabel('n');
    # plt.xlabel('Epsilon distance');
    # plt.show()
    # exit()
    eps_vals = np.arange(1, 10, 0.5)
    min_vals = range(10, 50)
    for eps_val in eps_vals:
        for min_sample in min_vals:
            clustering = DBSCAN(eps = eps_val, min_samples = min_sample).fit(complete_data)
            labels = clustering.labels_
            clusters =len(set(labels))-(1 if -1 in labels else 0)
            print(clusters,eps_val,min_sample)

def cluster_with_params(x1, x2, eps, min_samples):
    complete_data = np.concatenate((x1, x2))
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(complete_data)
    labels = clustering.labels_
    clusters =len(set(labels))-(1 if -1 in labels else 0)
    return labels, clusters

def brc_clustering(x1, x2):
    complete_data = np.concatenate((x1, x2))
    brc = Birch(n_clusters=5).fit(complete_data)
    labels = brc.labels_
    clusters =len(set(labels))-(1 if -1 in labels else 0)
    return labels, clusters


def main():
    random.seed(420)
    data_path = f'../data/Genes'
    print('reading data...')

    #------------Data Analysis------------#
    column_names = [f'gene_{i}' for i in range(0, 200)].append('Unnamed: 0')
    x = pd.read_csv(f'{data_path}/data.csv').set_index('Unnamed: 0')
    # x = pd.read_csv(f'{data_path}/data.csv',usecols=column_names).set_index('Unnamed: 0')
    y = pd.read_csv(f'{data_path}/labels.csv').set_index('Unnamed: 0').Class
    row_size, col_size = x.shape
    unique_classes = list(set(np.array(y.values)))
    print(f'number of rows: {row_size}\nnumber of columns: {col_size}')
    print(f'unique classes: {unique_classes}')
    # visualise_data(x, y, save=True, dims=2, title='visualisation of numeric data 2D')
    # visualise_data(x, y, save=True, dims=3, title='visualisation of numeric data 3D')


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=69)

    #------------Feature Extraction------------#
    print('\npreprocessing data...')
    pca = sklearnPCA(random_state=69)
    lda = LDA()
    KNN = KNeighborsClassifier(n_neighbors=5)
    clf = LogisticRegression(random_state=69, max_iter=200, solver='liblinear')
    # pca_components = determine_best_col_num(pca, clf, x_train.to_numpy(), y_train, 50, plot='PCA F1 Scores')
    # lda_components = determine_best_col_num(lda, clf, x_train.to_numpy(), y_train, 5, plot='LDA F1 Scores')
    # pca.set_params(n_components=pca_components)
    # lda.set_params(n_components=lda_components)

    #from previous experiments we found
    pca.set_params(n_components=10)
    lda.set_params(n_components=4)

    pca_x_train, pca_x_test = preprocess_data(pca, x_train, x_test, y_train)
    lda_x_train, lda_x_test = preprocess_data(lda, x_train, x_test, y_train)
    print(f'columns before PCA:{x_train.shape[1]}, columns after PCA:{pca_x_train.shape[1]}')
    print(f'columns before LDA:{x_train.shape[1]}, columns after LDA:{lda_x_train.shape[1]}')

    print('\ncomparing reduced data on KNN')
    original_train = x_train.to_numpy()
    original_test = x_test.to_numpy()

    original_pred = run_model(KNN, original_train, original_test, y_train)
    pca_pred = run_model(KNN, pca_x_train, pca_x_test, y_train)
    lda_pred = run_model(KNN, lda_x_train, lda_x_test, y_train)
    original_accuracy, original_f1 = get_score(original_pred, y_test)
    pca_accuracy, pca_f1 = get_score(pca_pred, y_test)
    lda_accuracy, lda_f1 = get_score(lda_pred, y_test)


    print(f'original accuracy: {original_accuracy}, original f1: {original_f1}')
    print(f'PCA accuracy: {pca_accuracy}, PCA f1: {pca_f1}')
    print(f'LDA accuracy: {lda_accuracy}, LDA f1: {lda_f1}')

    if lda_f1 >= pca_f1:
        reduced_train = lda_x_train
        reduced_test = lda_x_test
        print('LDA is the better reducer for this dataset')
    else:
        reduced_train = pca_x_train
        reduced_test = pca_x_test
        print('PCA is the better reducer for this dataset')


    #------------Classification------------#
    # print('\ntesting models...')
    # print('original dataset')
    # classify_data(original_train, original_test, y_train, y_test)
    # print('reduced data')
    # classify_data(reduced_train, reduced_test, y_train, y_test)

    #------------Ensemble------------#
    ###print(x.shape, '\n', y.shape)
    # print('\nrunning ensemble')
    # original_combined_data = np.concatenate((reduced_train, reduced_test))
    # original_combined_labels = np.concatenate((y_train, y_test))
    # ensemble(x, y)
    

    #------------Clustering------------#
    # print('\nclustering data...')
    # cluster_data_search(original_train, original_test)
    # cluster_data_search(reduced_train, reduced_test)
    # original_labels, original_clusters = brc_clustering(original_train, original_test)
    # print(original_labels)
    # reduced_labels, reduced_clusters = cluster_with_params(reduced_train, reduced_test, 3, 20)

    # print(f'Estimated no. of clusters for original dataset: {original_clusters}')
    # print(f'Estimated no. of clusters for reduced dataset: {reduced_clusters}')
    # original_data = np.concatenate((original_train, original_test))
    # reduced_data = np.concatenate((reduced_train, reduced_test))
    # visualise_cluster(original_data, original_labels, dims=2, save=True,  title='Clustering with Original Data 2D')
    # visualise_cluster(original_data, original_labels, dims=3, save=True,  title='Clustering with Original Data 3D')
    # visualise_cluster(reduced_data, reduced_labels, dims=2, save=True,  title='Clustering with Reduced Data 2D')
    # visualise_cluster(reduced_data, reduced_labels, dims=3, save=True,  title='Clustering with Reduced Data 3D')


    #------------Grid Search------------#

    #######---knn for original---#######
    # criterions = ['gini', 'entropy']
    # splitters = ['random', 'best']
    # max_features = [None, 'sqrt', 'log2']
    # original_settings = {}
    # original_best = 0
    # for criterion in tqdm(criterions, desc='criterion', leave=False):
    #     for splitter in tqdm(splitters, desc='splitters', leave=False):
    #         for max_feature in tqdm(max_features, desc='algorithms', leave=False):
    #             tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_features=max_feature, random_state=69)
    #             scores = cross_validate(tree, original_train, y_train, cv=10, return_train_score=True)
    #             ave_score = np.mean(scores['test_score'])

    #             if ave_score > original_best:
    #                 original_best = ave_score
    #                 original_settings = {
    #                     'criterion': criterion,
    #                     'splitter': splitter,
    #                     'max_features': max_feature,
    #                     'accuracy': original_best,
    #                     }
    # with open('../plots/numeric/original_settings.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    #     w = csv.DictWriter(f, original_settings.keys())
    #     w.writeheader()
    #     w.writerow(original_settings)

    #######---logregression for reduced---#######
    # solvers = ['newton-cg', 'lbfgs', 'liblinear']
    # penalties = ['none', 'l1', 'l2', 'elasticnet']
    # C_lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

    # reduced_settings = {}
    # reduced_best = 0
    # for solver in tqdm(solvers, desc='solvers', leave=False):
    #     for penalty in tqdm(penalties, desc='penalties', leave=False):
    #         for C_lr in tqdm(C_lrs, desc='C_lrs', leave=False):
    #             log = LogisticRegression(max_iter=400, solver=solver, C = C_lr, penalty = penalty, random_state=69)
    #             scores = cross_validate(log, reduced_train, y_train, cv=10, return_train_score=True)
    #             ave_score = np.mean(scores['test_score'])

    #             if ave_score > reduced_best:
    #                 reduced_best = ave_score
    #                 reduced_settings = {
    #                     'solver': solver,
    #                     'penalty': penalty,
    #                     'C_lr': C_lr,
    #                     'accuracy': reduced_best,
    #                     }
    #                 print(reduced_settings)
    # with open('../plots/numeric/reduced_settings_lr.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    #     w = csv.DictWriter(f, reduced_settings.keys())
    #     w.writeheader()
    #     w.writerow(reduced_settings)

    #------------Final Classification------------#
    print('\ntesting models...')
    print('original dataset')
    final_classify_data(x, y)
    print('reduced data')
    final_classify_data(x, y)






if __name__ == '__main__':
    main()
