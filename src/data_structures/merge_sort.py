## ------------------------------
## MergeSort

from heapq import merge


class MergeSort:
    ...


def merge(data, start, mid, end):
    ## build a temporary array to avoid modifying the original array
    tmp = []
    i, j = start, mid + 1
    k = 0

    while i <= mid and j <= end:
        if data[i] < data[j]:
            ## put ith data in temp
            tmp[k] = data[i]
            i += 1
            k += 1
        elif data[i] >= data[j]:
            ## put jth data in temp
            tmp[j] = data[j]
            j += 1
            k += 1

    while i <= mid:
        ## add the rest of left subarray into the temp array
        tmp[k] = data[i]
        i += 1
        k += 1

    while j <= end:
        tmp[k] = data[j]
        j += 1
        k += 1

    return tmp


def merge_sort(data, start, end):
    ## base case
    if start < end:
        mid = (start + end) // 2
        merge_sort(data, start, mid)
        merge_sort(data, mid + 1, end)
        merge(data, start, mid, end)
