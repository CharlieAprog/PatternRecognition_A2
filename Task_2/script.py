import pandas as pd
from sklearn import svm
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split

def mean(list):
    return sum(list) / len(list)

df = pd.read_csv('../Data/credit_cards/creditcard.csv')
df = df.drop(['Time', 'Amount'], 1)

print(df)

X = df.drop('Class', 1)
y = df.Class

n_reps = 10
labeled_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

per_lr = {}
for labeled_ratio in labeled_ratios:
    per_approach = {}
    for approach in ['semisupervised', 'supervised']:
        ratio_corrects = [-1 for _ in range(n_reps)]
        for i in range(n_reps):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle= True, stratify= True)
            if approach == 'semisupervised':
                X_train_lab, X_train_unlab, y_train_lab = X_train[:round(labeled_ratio * len(X_train))], X_train[round(labeled_ratio * len(X_train)):], y_train[:round(labeled_ratio * len(y_train))]
                print(y_train_lab)
                label_prop_model = LabelPropagation()
                label_prop_model.fit(X_train_lab + X_train_unlab, y_train_lab + [-1 for _ in X_train_unlab])
                y_train_unlab = label_prop_model.predict(X_train_unlab)
                y_train = y_train_lab + y_train_unlab

            predictor = svm.SVC()
            predictor.fit(X_train, y_train)
            predicted = predictor.predict(X_test)
            ratio_correct = mean([predicted[i] == y_test[i] for i in range(len(predicted))])
            ratio_corrects[i] = ratio_correct
        per_approach[approach] = mean(ratio_corrects)
    per_lr[labeled_ratio] = per_approach

for lr, pa in per_lr.items():
    print(lr, pa)