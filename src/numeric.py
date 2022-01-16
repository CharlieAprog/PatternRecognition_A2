import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def feature_selection(x_train, x_test, y_train):
    pass

def process_data(model, x_train, x_test, y_train):
    x_train = model.fit_transform(x_train, y_train)
    x_test = model.transform(x_test)
    return x_train, x_test

def visualise_data(x, y):
    title = 'TSNE plot for numeric data'
    pca = sklearnPCA(n_components=2) #2-dimensional PCA
    pca_transformed = pd.DataFrame(pca.fit_transform(x))
    pca_transformed.index = y.index

    lda = LDA(n_components=2) #2-dimensional LDA
    lda_transformed = pd.DataFrame(lda.fit_transform(x, y))
    lda_transformed.index = y.index

    make_tsne_plot(pca_transformed, y,f'{title}_PCA', save_plot=True)
    make_tsne_plot(lda_transformed, y,f'{title}_LDA', save_plot=True)


def make_tsne_plot(data, labels, title, save_plot=False):
    classes = list(set(np.array(labels.values)))
    for class_ in classes:
        plt.scatter(data[labels==class_][0], data[labels==class_][1], label=class_)

    # Display legend and show plot
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
    print(row_size, col_size)
    unique_classes = list(set(np.array(y.values)))
    print(unique_classes)
    visualise_data(x, y)

    pass

if __name__ == '__main__':
    main()
