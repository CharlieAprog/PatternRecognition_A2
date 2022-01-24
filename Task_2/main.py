import random

def label_unlabeled(labeled_data, unlabeled_data, clustering_algorithm, supervised_learning_algorithm):
    # Input: labeled_data ((x1, y1), ..., (xl, yl),)
    #        unlabeled_data (x(l+1), ..., x(l+u))
    #        clustering_algorithm (xs -> clusters ([[cluster_1_idx_1, cluster_1_idx_2, ...], [cluster_2_idx_1, ...], ...]))
    #        supervised_learning_algorithm (training_data -> predictor_object)

    # Output: labels for the unlabeled_data

    labels_unlabeled_data = [-1 for _ in unlabeled_data]
    xs = [x for (x, _) in labeled_data] + unlabeled_data
    clusters = clustering_algorithm(xs)
    for cluster in clusters:
        labeled_instances = [labeled_data[idx] for idx in cluster if idx < len(labeled_data)]
        supervised_predictor = supervised_learning_algorithm(labeled_instances)
        unlabeled_instance_indices = [idx - len(labeled_data) for idx in cluster if idx >= len(labeled_data)]
        unlabeled_instances = [unlabeled_data[idx] for idx in unlabeled_instance_indices]
        labels = supervised_predictor(unlabeled_instances)
        for i, label in enumerate(labels):
            idx = unlabeled_instance_indices[i]
            labels_unlabeled_data[idx] = label

    return labels_unlabeled_data

def semi_supervised_learning(labeled_data, unlabeled_data, clustering_algorithm, supervised_learning_algorithm):
    labels_unlabeled_data = label_unlabeled(labeled_data, unlabeled_data, clustering_algorithm, supervised_learning_algorithm)
    newly_labeled_data = [(unlabeled_data[i], labels_unlabeled_data[i]) for i in range(len(unlabeled_data))]
    training_data = labeled_data + newly_labeled_data
    predictor = supervised_learning_algorithm(training_data)
    return predictor

def supervised_learning(labeled_data, supervised_learning_algorithm):
    predictor = supervised_learning_algorithm(labeled_data)
    return predictor

def prepare_data():
    pass

def read_data(file_path)
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data_per_class = {}

    for line in lines[1:]:
        row = line.strip().split(',')
        input_vector = row[1:29]
        target = row[-1]
        if target not in data_per_class.keys():
            data_per_class[target] = []
        data_per_class[target].append((input_vector, target))

    return data_per_class

def balance_data(data_per_class):
    n_data_points_per_class = [len(dpc) for dpc in data_per_class]
    min_n_data_points = min(n_data_points_per_class)
    return [dpc[:min_n_data_points] for dpc in data_per_class]

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

def full_script():
    data_per_class = read_data('../Data/credit_cards/creditcard.csv')
    balanced_data_per_class = balance_data(data_per_class)
    labeled_train_data, unlabeled_train_data, test_data = [], [], []
    for _, bdpc in balanced_data_per_class.items():
        train_data_per_class, test_data_per_class = split_data(bdpc, [0.8, 0.2])
        labeled_train_data_per_class, unlabeled_train_data_per_class = split_data(train_data_per_class, [0.3, 0.7])
        labeled_train_data, unlabeled_train_data, test_data += labeled_train_data_per_class, unlabeled_train_data_per_class, test_data_per_class
    predictor = semi_supervised_learning(labeled_train_data, [x for (x, _) in unlabeled_train_data], clustering_algorithm, supervised_learning_algorithm)
    
