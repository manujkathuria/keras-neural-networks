import numpy as np

arr = [20, 2, 7, 1, 34]
np.mean(arr)

# 1D array
arr = [20, 2, 7, 1, 34]

print("arr : ", arr)
print("std of arr : ", np.std(arr))

import numpy as np

# 2D array
arr = [[2, 2, 2, 2, 2],
       [15, 6, 27, 8, 2],
       [23, 2, 54, 1, 2, ],
       [11, 44, 34, 7, 2]]

# std of the flattened array
print("\nstd of arr, axis = None : ", np.std(arr))

# axis = 0 means SD along the column and axis = 1 means SD along the row.

# std along the axis = 0
print("\nstd of arr, axis = 0 : ", np.std(arr, axis=0))

# std along the axis = 1
print("\nstd of arr, axis = 1 : ", np.std(arr, axis=1))