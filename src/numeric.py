import random
import math
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def make_scatter_plot(data, labels, title, save_plot=False, dimension=2, cluster=None):
    if not any(cluster):
        classes = list(set(np.array(labels.values)))
    fig = plt.figure()
    if dimension == 3:
        ax = fig.add_subplot(projection='3d')
        if any(cluster):
            ax.scatter(data[0], data[1], data[2], c=cluster)
        else:
            for class_ in classes:
                ax.scatter(data[labels==class_][0], data[labels==class_][1], data[labels==class_][2], label=class_)
    else:
        ax = fig.add_subplot()
        if any(cluster):
            ax.scatter(data[0], data[1], c=cluster)
        else:
            for class_ in classes:
                ax.scatter(data[labels==class_][0], data[labels==class_][1], label=class_)

    plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(f'../plots/numeric/{title}.jpg')
    else:
        plt.show()

def visualise_data(x, y, dims=2, save=False):
    title = 'visualisation of numeric data'
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

def get_score(pred, y_test):
    accuracy = round(accuracy_score(y_test, pred), 3)
    f1 = round(f1_score(y_test, pred, average="macro", zero_division=0), 3)
    return accuracy, f1

def calculate_kn_distance(X,k):

    kn_distance = []
    for i in range(len(X)):
        eucl_dist = []
        for j in range(len(X)):
            eucl_dist.append(
                math.sqrt(
                    ((X[i,0] - X[j,0]) ** 2) +
                    ((X[i,1] - X[j,1]) ** 2)))

        eucl_dist.sort()
        kn_distance.append(eucl_dist[k])

    return kn_distance


def cluster_data_search(x_train, x_test):
    complete_data = np.concatenate((x_train, x_test))
    norm = (complete_data - np.min(complete_data))/np.ptp(complete_data)

    print(np.max(norm), np.min(norm))
    # eps_dist = calculate_kn_distance(norm,4)
    # plt.hist(eps_dist,bins=30)
    # plt.ylabel('n');
    # plt.xlabel('Epsilon distance');
    # plt.show()

    eps_vals = np.arange(0.0001, 0.006, 0.0001)
    min_vals = range(3, 6)
    for eps_val in eps_vals:
        for min_sample in min_vals:
            clustering = DBSCAN(eps = eps_val, min_samples = min_sample).fit(norm)
            labels = clustering.labels_
            clusters =len(set(labels))-(1 if -1 in labels else 0)
            print(clusters,eps_val,min_sample)

def cluster_with_params(x1, x2, eps, min_samples):
    complete_data = np.concatenate((x1, x2))
    clustering = DBSCAN(eps = eps, min_samples = min_samples).fit(complete_data)
    labels = clustering.labels_
    clusters =len(set(labels))-(1 if -1 in labels else 0)
    return labels, clusters

def k_means_clustering(x1, x2):
    complete_data = np.concatenate((x1, x2))
    kmeans = KMeans(n_clusters=5, random_state=69).fit(complete_data)
    labels = kmeans.labels_
    clusters =len(set(labels))-(1 if -1 in labels else 0)
    return labels, clusters


def main():
    random.seed(420)
    data_path = f'../data/Genes/Original'
    print('reading data...')

    #------------Data Analysis------------#
    column_names = [f'gene_{i}' for i in range(0, 200)]
    column_names.append('Unnamed: 0')
    x = pd.read_csv(f'{data_path}/data.csv').set_index('Unnamed: 0')
    # x = pd.read_csv(f'{data_path}/data.csv',usecols=column_names).set_index('Unnamed: 0')
    y = pd.read_csv(f'{data_path}/labels.csv').set_index('Unnamed: 0').Class
    row_size, col_size = x.shape
    unique_classes = list(set(np.array(y.values)))
    print(f'number of rows: {row_size}\nnumber of columns: {col_size}')
    print(f'unique classes: {unique_classes}')
    # visualise_data(x, y, save=False, dimensions=3)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=69)

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


    #------------Clustering------------#
    print('\nclustering data...')
    # cluster_data_search(original_train, original_test)
    # cluster_data_search(reduced_train, reduced_test)
    original_labels, original_clusters = k_means_clustering(original_train, original_test)
    print(original_labels)
    # reduced_labels, reduced_clusters = cluster_with_params(reduced_train, reduced_test, 5.63, 7)

    print(f'Estimated no. of clusters for original dataset: {original_clusters}')
    # print(f'Estimated no. of clusters for reduced dataset: {reduced_clusters}')
    original_data = np.concatenate((reduced_train, reduced_test))
    # reduced_data = np.concatenate((reduced_train, reduced_test))
    visualise_cluster(original_data, original_labels, dims=2, save=True,  title='Clustering with Original Data 2D')
    visualise_cluster(original_data, original_labels, dims=3, save=True,  title='Clustering with Original Data 3D')
    # visualise_cluster(reduced_data, reduced_labels, dims=2, save=True,  title='Clustering with Reduced Data 2D')
    # visualise_cluster(reduced_data, reduced_labels, dims=3, save=True,  title='Clustering with Reduced Data 3D')
    exit()
    #------------Grid Search------------#
    # #knn for original
    # neighbors  = range(1,31)
    # weights = ['uniform', 'distance']
    # algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
    # ps  = [1, 2]
    # original_settings = {}
    # original_best = 0
    # for weight in tqdm(weights, desc='weights', leave=False):
    #     for p in tqdm(ps, desc='ps', leave=False):
    #         for algorithm in tqdm(algorithms, desc='algorithms', leave=False):
    #             for neighbor in tqdm(neighbors, desc='neighbors', leave=False):
    #                 KNN = KNeighborsClassifier(n_neighbors=neighbor, weights=weight, p=p, algorithm=algorithm)
    #                 scores = cross_validate(KNN, original_train, y_train, cv=10, scoring=make_scorer(f1_score, average='weighted') , return_train_score=True)
    #                 ave_score = np.mean(scores['test_score'])

    #                 if ave_score > original_best:
    #                     original_best = ave_score
    #                     original_settings = {
    #                         'weights': weight,
    #                         'p': p,
    #                         'algorithm': algorithm,
    #                         'neightbor': neighbor,
    #                         'f1_score': original_best,
    #                         }
    # with open('../plots/numeric/original_settings.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    #     w = csv.DictWriter(f, original_settings.keys())
    #     w.writeheader()
    #     w.writerow(original_settings)



    #logregression for reduced

    #------------Ensemble------------#





if __name__ == '__main__':
    main()
