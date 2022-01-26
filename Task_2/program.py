import random
from sklearn import svm
from sklearn.semi_supervised import LabelPropagation


from clustering_algos import euclidean_distance, propagating_1_nn

def fit_svm(inputs, targets):
    predictor = svm.SVC()
    predictor.fit(inputs, targets)
    return predictor

def semi_supervised_learning(labeled_data, unlabeled_data, supervised_learning_algorithm):
    # training_data = propagating_1_nn(labeled_data, unlabeled_data)
    data = [x for x, _ in labeled_data] + unlabeled_data
    labels = [y for _, y in labeled_data] + [-1 for _ in unlabeled_data]
    label_prop_model = LabelPropagation()
    label_prop_model.fit(data, labels)
    unlabeled_labels = label_prop_model.predict(unlabeled_data)
    training_data = labeled_data + [(unlabeled_data[i], unlabeled_labels[i]) for i in range(len(unlabeled_data))]
    predictor = supervised_learning(training_data, supervised_learning_algorithm)
    return predictor

def supervised_learning(labeled_data, supervised_learning_algorithm):
    inputs, targets = split_inputs_targets(labeled_data)
    predictor = supervised_learning_algorithm(inputs, targets)
    return predictor

def split_inputs_targets(data):
    inputs = [x for x, _ in data]
    targets = [y for _, y in data]    
    return inputs, targets

def read_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_per_class = {}

    for line in lines[1:]:
        row = line.strip().split(',')
        input_vector = [float(c) for c in row[1:29]]
        target = int(row[-1][1:-1])
        if target not in data_per_class.keys():
            data_per_class[target] = []
        data_per_class[target].append((input_vector, target))

    return data_per_class

def balance_data(data_per_class):
    n_data_points_per_class = [len(dpc) for dpc in data_per_class.values()]
    min_n_data_points = min(n_data_points_per_class)
    return [dpc[:min_n_data_points] for dpc in data_per_class.values()]

def split_data(data, ratios):
    # Input: data ([(x1, y1), (x2, y2), ...]
    #        ratios ([ratio1, ratio2, ...])

    # Output: s_data ([[(group_1_x_1, group_1_y_1), (group_1_x_2, group_1_y_2), ...], [(group_2_x_1, group_2_y_1), ...], ...])

    random.shuffle(data)
    s_data = [[] for i in range(len(ratios))]
    end_idx = 0
    for i in range(len(ratios)):
        start_idx = end_idx
        end_idx = end_idx + round(ratios[i] * len(data))
        s_data[i] = data[start_idx:end_idx]
    return s_data

def test_performance(predictor, test_data, performance_test):
    inputs, targets = split_inputs_targets(test_data)
    classifications = predictor.predict(inputs)
    performance_results = performance_test(targets, classifications)
    return performance_results

def prepare_data(data_per_class, test_split, unlabled_split):
    balanced_data_per_class = balance_data(data_per_class)
    labeled_train_data, unlabeled_train_data, test_data = [], [], []
    for bdpc in balanced_data_per_class:
        train_data_per_class, test_data_per_class = split_data(bdpc, test_split)
        labeled_train_data_per_class, unlabeled_train_data_per_class = split_data(train_data_per_class, unlabled_split)
        labeled_train_data += labeled_train_data_per_class
        unlabeled_train_data += unlabeled_train_data_per_class
        test_data += test_data_per_class
    random.shuffle(labeled_train_data)
    random.shuffle(unlabeled_train_data)
    random.shuffle(test_data)
    return labeled_train_data, unlabeled_train_data, test_data

def check_if_correct(output, target):
    if len(output) != len(target):
        print('Output, target lengths do not match', len(output), len(target))
        exit(-1)
    return [output[i] == target[i] for i in range(len(output))]

def total_correct(output, target):
    return sum(check_if_correct(output, target))

def mean_correct(output, target):
    tc = total_correct(output, target)
    return tc / len(output)

def mean(list):
    return sum(list) / len(list)

n_reps = 3

def full_script():
    data_per_class = read_data('../Data/credit_cards/creditcard.csv')

    test_ratio = 0.2
    unlabeled_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    results = []

    for unlabeled_ratio in unlabeled_ratios:
        print(unlabeled_ratio)

        test_split = [1 - test_ratio, test_ratio]
        unlabeled_split = [1 - unlabeled_ratio, unlabeled_ratio]

        semi_sup_perf = []
        for i in range(n_reps):
            labeled_train_data, unlabeled_train_data, test_data = prepare_data(data_per_class, test_split, unlabeled_split)
            predictor = semi_supervised_learning(labeled_train_data, [x for (x, _) in unlabeled_train_data], fit_svm)
            performance = test_performance(predictor, test_data, mean_correct)
            semi_sup_perf.append(performance)
        
        sup_perf = []
        for _ in range(n_reps):
            labeled_train_data, unlabeled_train_data, test_data = prepare_data(data_per_class, test_split, unlabeled_split)
            predictor = supervised_learning(labeled_train_data, fit_svm)
            performance = test_performance(predictor, test_data, mean_correct)
            sup_perf.append(performance)

        results.append((unlabeled_ratio, mean(semi_sup_perf), mean(sup_perf)))

    for r in results:
        print(r)
    

full_script()