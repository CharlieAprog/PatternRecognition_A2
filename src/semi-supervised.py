from cv2 import Subdiv2D_PREV_AROUND_DST
import pandas as pd
import numpy as np
from tqdm import tqdm


from sklearn import svm
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

def plot_it(res):
    semisuper = [r['semisupervised'][0] for r in res.values()]
    super_ = [r['supervised'][0] for r in res.values()]
    jlp = [r['justlabelpropagation'][0] for r in res.values()]
    ratio_labeled = [r for r in res.keys()]

    plt.plot(ratio_labeled, jlp, label= 'Label Propagation Only')
    plt.plot(ratio_labeled, semisuper, label= 'Semi-supervised')
    plt.plot(ratio_labeled, super_, label= 'Supervised')
    plt.legend(loc='lower left')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Ratio of Labeled Data')
    plt.ylabel('Percentage Correct')
    plt.savefig(f'../plots/semi-supervised/semi_supervised_plot.jpg')
    plt.show()

def main():
    np.seterr(invalid='ignore')
    df = pd.read_csv('../data/creditcard.csv')
    X = df.drop(columns=['Time', 'Amount', 'Class'], axis=1).values
    y = df['Class'].values

    n_reps = 10
    labeled_ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]

    per_lr = {}
    for labeled_ratio in tqdm(labeled_ratios, desc='different ratios', leave=False):
        per_approach = {}
        for approach in tqdm(['semisupervised', 'supervised', 'justlabelpropagation'], desc='learning type', leave=False):
            # print(labeled_ratio, approach)
            ratio_corrects = [-1 for _ in range(n_reps)]
            ratio_corrects_lab_prop = [-1 for _ in range(n_reps)]
            for i in range(n_reps):
                under_sampler = RandomUnderSampler()
                X_balanced, y_balanced = under_sampler.fit_resample(X, y)
                X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size= 0.2, shuffle= True)

                if approach == 'semisupervised' or approach == 'justlabelpropagation':
                    X_train_lab, X_train_unlab, y_train_lab = X_train[:round(labeled_ratio * len(X_train))], X_train[round(labeled_ratio * len(X_train)):], y_train[:round(labeled_ratio * len(y_train))]
                    label_prop_model = LabelPropagation()
                    label_prop_model.fit(X_train, np.append(y_train_lab, [-1 for _ in X_train_unlab]))

                if approach == 'justpropagation':
                    predicted = label_prop_model.predict(X_test)

                if approach == 'semisupervised':
                    y_train_unlab = label_prop_model.predict(X_train_unlab)
                    y_train = np.append(y_train_lab, y_train_unlab)

                if approach == 'supervised' or approach == 'semisupervised':
                    predictor = svm.SVC()
                    predictor.fit(X_train, y_train)
                    predicted = predictor.predict(X_test)

                ratio_correct = np.mean([predicted[i] == y_test[i] for i in range(len(predicted))])
                ratio_corrects[i] = ratio_correct

            per_approach[approach] = (np.mean(ratio_corrects), np.std(ratio_corrects), np.std(ratio_corrects) / np.sqrt(10))
        per_lr[labeled_ratio] = per_approach

    # prints mean, standard deviation, and standard error for the success rates.
    # for lr, pa in per_lr.items():
    #     print(lr, pa)

    # plot_it(per_lr)

if __name__ == '__main__':
    main()
