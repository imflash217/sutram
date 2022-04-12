# import requests
# import mysql.connector
# import pandas as pd
"""
Given a collection of vectors and k, write a function to return the mean vector of the k nearest neighbors of each vector in the collection.
Example 1:
  Input: [[0],[2]], k = 1
  Output: [[2],[0]]
Example 2:
  Input: [[0,0],[0,1],[1,0],[0,2],[2,0]], k = 2
  Output: [[0.5,0.5],[0,1],[1,0],[0,0.5],[0.5,0]]
"""


def dist(v1, v2):
    # out = sum((v1-v2)**2)
    out = 0
    for a, b in zip(v1, v2):
        out += (a - b) ** 2
    # print(out, v1, v2)
    return out


def find_k_means(vectors, k):
    ##
    result = []
    N = len(vectors)
    for i, v in enumerate(vectors):
        # pivot
        curr = []
        for j, u in enumerate(vectors):
            if i != j:
                dis = dist(v, u)
                curr.append((j, dis))
        curr = sorted(curr, key=lambda x: x[1])
        top_k_idxs = [x[0] for x in curr[:k]]

        ## find mean of top k
        mean = [0] * len(v)
        for q in top_k_idxs:
            for p in range(len(mean)):
                mean[p] += vectors[q][p]

        mean = [m / k for m in mean]
        result.append(mean)

    return result


def main():
    input_v = [[0, 0], [0, 1], [1, 0], [0, 2], [2, 0]]
    k = 2
    result = find_k_means(input_v, k)
    print(result)
    assert result == [[0.5, 0.5], [0, 1], [1, 0], [0, 0.5], [0.5, 0]]


main()
