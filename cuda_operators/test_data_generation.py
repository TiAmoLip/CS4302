import numpy as np

# A = np.fromfile('A.bin', dtype=np.float32).reshape(1, 1280)
# B = np.fromfile('B.bin', dtype=np.float32).reshape(1280, 5893)
# C = np.fromfile('C.bin', dtype=np.float32).reshape(1, 5893)

# new_A = B.T
# new_B = A.T
# new_C = new_A @ new_B

# new_A.tofile('new_A.bin')
# new_B.tofile('new_B.bin')
# new_C.tofile('new_C.bin')

A = np.random.randn(5893, 1280).astype(np.float32)
B = np.random.randn(1280, 1).astype(np.float32)
C = np.dot(A, B)

A.tofile('A.bin')
B.tofile('B.bin')
C.tofile('C.bin')