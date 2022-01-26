from heapq import heappush, heappop
import numpy as np

def euclidean_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    p = np.sum((x-y)**2)
    d = np.sqrt(p)
    return d

def propagating_1_nn(labeled_data, unlabeled_data, distance_function = euclidean_distance):
    new_labeled_data_set = labeled_data + [None for _ in unlabeled_data]

    L = list(range(len(labeled_data)))
    U = [u + len(labeled_data) for u in list(range(len(unlabeled_data)))]

    distance_heap = []

    for l in L:
        for u in U:
            distance = distance_function(labeled_data[l][0], unlabeled_data[u-len(labeled_data)])
            heap_tuple = (distance, (l, u))
            heappush(distance_heap, heap_tuple)
    i = 0
    while U:
        u = -1
        while u not in U:
            (distance, (l, u)) = heappop(distance_heap)
        vector = unlabeled_data[u - len(labeled_data)]
        label = new_labeled_data_set[l][1]
        new_labeled_data_set[u] = (vector, label)
        U.remove(u)
        l = u
        for u in U:
            distance = distance_function(vector, unlabeled_data[u-len(labeled_data)])
            heap_tuple = (distance, (l, u))
            heappush(distance_heap, heap_tuple)
    return new_labeled_data_set

labeled_data = [([0, 0], 0), ([100, 100], 1), ([200,0], 2)]
unlabeled_data = [[1,3], [2, 2], [-1, -1], [-1, -1], [200, 1], [200, 200], [50, 50], [60, 60]]
new_labeled_data = propagating_1_nn(labeled_data, unlabeled_data)
for nld in new_labeled_data:
    print(nld)
for vec, tar in new_labeled_data:
    print('{}\t{}'.format(vec, tar))
