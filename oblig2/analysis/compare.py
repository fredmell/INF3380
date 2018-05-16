import numpy as np
import struct

class Matrix:
    def __init__(self, filename):
        self.filename = filename
        self.read_matrix()

    def read_matrix(self):
        f = open(self.filename, "rb")
        num_rows, = struct.unpack('i', f.read(4))
        num_cols, = struct.unpack('i', f.read(4))

        self.arr = np.fromfile(f, dtype=np.float64)
        self.arr = np.reshape(self.arr, (num_rows, num_cols))

fn_1 = "../data/output/small_matrix_c.bin"
fn_2 = "../data/solution/small_matrix_c.bin"

mat1 = Matrix(fn_1)
mat2 = Matrix(fn_2)

print(mat1.arr[10,:], "\n" ,mat2.arr[10,:])
# print(np.array_equal(mat1.arr, mat2.arr))
