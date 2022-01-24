def knn(labeled_data, unlabeled_data, setup):
    k = setup['k']

    for h, (ul_vector, _) in unlabeled_data:
        distance_vector = [0 for _ in labeled_data]
        for i, (l_vector, _) in enumerate(labeled_data):
            distance = (sum([(ul_vector[j] - l_vector[j]) ^ 2 for j in range(len(ul_vector))]))^(0.5)
            distance_vector[i] = (distance, i)
        distance_vector.sort(key=lambda x: x[0])
        nearest_neighbours = distance_vector[:k]
        labels_nearest_neighbors = [unlabeled_data[i]['target'] for _, i in nearest_neighbours]
        label = max(set(labels_nearest_neighbors), key=labels_nearest_neighbors.count)
        unlabeled_data[h] = (ul_vector, label)

    return unlabeled_data


def kmeans(labeled_data, unlabeled_data):
    # returns unlabeled_data but now it's labeled

    # create 
    pass