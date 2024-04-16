'''
import numpy as np

matrix = np.array([[1, 0, 3],
                   [4, 5, 6],
                   [7, 9, 0]])
#matrix = np.zeros((3,4))

# Value to find
value_to_find = 0

# Find indices where value equals value_to_find
indices = np.where(matrix == value_to_find)

print(matrix[1,0])

matrix[1,0] = 10

print(matrix)

print(indices[0], indices[1])

if len(indices[0]) > 0:
    # Print the indices of the value
    print(f"The value {value_to_find} is located at indices:")
    for i in range(len(indices[0])):
        print(f"({indices[0][i]}, {indices[1][i]})")
else:
    print(f"The value {value_to_find} is not found in the matrix.")
    '''

a = 2**(1) * 15_000
print(a)