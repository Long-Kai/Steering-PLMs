import numpy as np

def refine_list(list1, list2):
    # return a new list of elements in list1 but not in list2
    list3 = [ele for ele in list1 if ele not in list2] 
    return list3

def count_overlap(list1, list2):
    # return number of elements both in list1 and list2
    acc = [ele in list2 for ele in list1]
    return np.sum(acc)

def list_intersection(list1, list2):
    # return a new list of elements both in list1 and list2
    list3 = [ele for ele in list1 if ele in list2] 
    return list3

def topk_intersection_indices(list1, list2, k, reverse=False):
    # return top k elements in the intersection of two sorted lists list1 and list2
    order = -1 if reverse else 1
    idx1 = np.argsort(list1)[::order]
    idx2 = np.argsort(list2)[::order]
    current_k = k
    inter = list_intersection(idx1[:current_k], idx2[:current_k])
    while len(inter) < k and current_k < len(list1):
        current_k += 1
        inter = list_intersection(idx1[:current_k], idx2[:current_k])
    return inter