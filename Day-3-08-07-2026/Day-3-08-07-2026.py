import numpy as np
 
arr = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])
 
print("Original array:")
print(arr)
print("Shape:", arr.shape)
print()
 
print("Element at [0,0]:", arr[0, 0])
print("Element at [1,2]:", arr[1, 2])
print("Element at [-1,-1]:", arr[-1, -1])
print("Row 1:", arr[1])
print("Column 2:", arr[:, 2])
print()
 
print("arr[0,0] + arr[0,1]:", arr[0, 0] + arr[0, 1])
print("arr[1,0] * arr[1,1]:", arr[1, 0] * arr[1, 1])
print("col3 - col0:", arr[:, 3] - arr[:, 0])
print("row2 / 10:", arr[2] / 10)
print()
 
print("1. Sum:", np.sum(arr))
print("2. Mean:", np.mean(arr))
print("3. Median:", np.median(arr))
print("4. Std:", np.std(arr))
print("5. Variance:", np.var(arr))
print("6. Max:", np.max(arr))
print("7. Min:", np.min(arr))
print("8. Argmax:", np.argmax(arr))
print("9. Argmin:", np.argmin(arr))
print("10. Reshape (4x3):\n", np.reshape(arr, (4, 3)))
print("11. Transpose:\n", np.transpose(arr))
print("12. Flatten:", np.ravel(arr))
print("13. Sqrt:\n", np.sqrt(arr))
print("14. Cumsum:", np.cumsum(arr))
print("15. Sort (axis=1):\n", np.sort(arr, axis=1))
print("16. Vstack:\n", np.vstack((arr, [[1, 2, 3, 4]])))
print("17. Split (3 parts):", np.split(arr, 3, axis=0))