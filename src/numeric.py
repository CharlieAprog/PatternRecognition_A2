import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, f1_score


def run_model(model, x_train, x_test, y_train):
    model.fit(x_train, y_train)
    return model.predict(x_test)

def classify_data(x_train, x_test, y_train, y_test):
    KNN = KNeighborsClassifier(n_neighbors=5)
    y_predKNN = run_model(KNN, x_train, x_test, y_train)
    knn_accuracy, knn_f1 = get_score(y_predKNN, y_test)

    Tree = DecisionTreeClassifier()
    y_predTree = run_model(Tree, x_train, x_test, y_train)
    tree_accuracy, tree_f1 = get_score(y_predTree, y_test)

    Forest = RandomForestClassifier()
    y_predForest = run_model(Forest, x_train, x_test, y_train)
    forest_accuracy, forest_f1 = get_score(y_predForest, y_test)

    print(f'knn accuracy:{knn_accuracy} knn f1:{knn_f1}')
    print(f'tree accuracy:{tree_accuracy} tree f1:{tree_f1}')
    print(f'forest accuracy:{forest_accuracy} forest f1:{forest_f1}')

def test_data_reductions(x_train, x_test, y_train, y_test):
    KNN = KNeighborsClassifier(n_neighbors=5)
    pca_pred = run_model(KNN, x_train, x_test, y_train)
    pca_accuracy = accuracy_score(y_test, pca_pred)
    pca_f1 = f1_score(y_test, pca_pred, average="macro", zero_division=0)

    lda_pred = run_model(KNN, x_train, x_test, y_train)
    lda_accuracy = accuracy_score(y_test, lda_pred)
    lda_f1 = f1_score(y_test, lda_pred, average="macro", zero_division=0)

    print(f'pca accuracy: {pca_accuracy}, pca f1: {pca_f1}')
    print(f'lda accuracy: {lda_accuracy}, lda f1: {lda_f1}')


def preprocess_data(model, x_train, x_test, y_train):
    x_train = model.fit_transform(x_train, y_train)
    x_test = model.transform(x_test)
    return x_train, x_test

def visualise_data(x, y, dims=2, save=False):
    title = 'visualisation of numeric data'
    pca = sklearnPCA(n_components=dims) #2-dimensional PCA
    pca_transformed = pd.DataFrame(pca.fit_transform(x))
    pca_transformed.index = y.index
    make_scatter_plot(pca_transformed, y,f'{title}_PCA', save_plot=save)

    lda = LDA(n_components=dims) #2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(x, y))
    lda_transformed.index = y.index
    make_scatter_plot(lda_transformed, y,f'{title}_LDA', save_plot=save)

def get_score(pred, y_test):
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro", zero_division=0)
    return accuracy, f1


def make_scatter_plot(data, labels, title, save_plot=False, dimension=2):
    classes = list(set(np.array(labels.values)))
    fig = plt.figure()
    if dimension == 3:
        ax = fig.add_subplot(projection='3d')
        for class_ in classes:
            ax.scatter(data[labels==class_][0], data[labels==class_][1], data[labels==class_][2], label=class_)
    else:
        ax = fig.add_subplot()
        for class_ in classes:
            ax.scatter(data[labels==class_][0], data[labels==class_][1], label=class_)

    plt.title(title)
    plt.legend()
    if save_plot:
        plt.savefig(f'../plots/numeric/{title}.jpg')
    else:
        plt.show()


def main():
    data_path = f'../data/Genes/Original'
    print('reading data...')
    x = pd.read_csv(f'{data_path}/data.csv').set_index('Unnamed: 0')
    y = pd.read_csv(f'{data_path}/labels.csv').set_index('Unnamed: 0').Class
    row_size, col_size = x.shape
    unique_classes = list(set(np.array(y.values)))
    print(f'number of rows: {row_size}\nnumber of columns: {col_size}')
    print(f'unique classes: {unique_classes}')
    # visualise_data(x, y, save=False, dimensions=3)
    x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, stratify=y
        )

    print('preprocessing data...')
    pca = sklearnPCA()
    lda = LDA()
    pca_x_train, pca_x_test = preprocess_data(pca, x_train, x_test, y_train)
    print(f'pca train before:{x_train.shape}, after:{pca_x_train.shape}')
    print(f'pca test before:{x_test.shape}, after:{pca_x_test.shape}')

    lda_x_train, lda_x_test = preprocess_data(pca, x_train, x_test, y_train)
    print(f'lda train before:{x_train.shape}, after:{lda_x_train.shape}')
    print(f'lda test before:{x_test.shape}, after:{lda_x_test.shape}')

    print('testing preprocessed data')
    KNN = KNeighborsClassifier(n_neighbors=5)
    original_pred = run_model(KNN, x_train, x_test, y_train)
    original_accuracy = accuracy_score(y_test, original_pred)
    original_f1 = f1_score(y_test, original_pred, average="macro", zero_division=0)

    pca_pred = run_model(KNN, pca_x_train, pca_x_test, y_train)
    pca_accuracy = accuracy_score(y_test, pca_pred)
    pca_f1 = f1_score(y_test, pca_pred, average="macro", zero_division=0)

    lda_pred = run_model(KNN, pca_x_train, pca_x_test, y_train)
    lda_accuracy = accuracy_score(y_test, lda_pred)
    lda_f1 = f1_score(y_test, lda_pred, average="macro", zero_division=0)

    print(f'original accuracy: {original_accuracy}, original f1: {original_f1}')
    print(f'pca accuracy: {pca_accuracy}, pca f1: {pca_f1}')
    print(f'lda accuracy: {lda_accuracy}, lda f1: {lda_f1}')

    print('testing models')
    classify_data(pca_x_train, pca_x_test, y_train, y_test)




if __name__ == '__main__':
    main()
