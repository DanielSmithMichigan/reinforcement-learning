import numpy as np
def getColumn(arr, column):
    out = []
    for i in range(len(arr)):
        out.append(arr[i][column])
    return out
def oneHot(arr, maxLen):
    out = []
    for i in range(len(arr)):
        entry = np.zeros(maxLen)
        entry[arr[i]] = 1
        out.append(entry)
    return out

