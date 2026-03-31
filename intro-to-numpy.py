import numpy as np

one_dimensional_arr = np.array([10,12])
print(one_dimensional_arr)

b = np.arange(3)
print(b)

c = np.arange(1,20,3)
print(c)

lin_spaced_arr = np.linspace(0,100,5)
print(lin_spaced_arr)

lin_spaced_arr_int = np.linspace(0,100,5,dtype=int)
print(lin_spaced_arr_int)

c_int = np.arange(1,20,3,dtype=int)
print(c_int)

b_float = np.arange(3, dtype=float)
print(b_float)


char_arr = np.array(['Welcome to Math for ML!'])
print(char_arr)
print(char_arr.dtype)

ones_arr = np.ones(3,dtype=int)
print(ones_arr)

zeros_arr = np.zeros(3)
print(zeros_arr)

empty_arr = np.empty(3)
print(empty_arr)

rand_arr = np.random.rand(3)
print(rand_arr)

two_dimensional_arr =   np.array([[1,2,3],[4,5,6]])
print(two_dimensional_arr)

one_dim_arr =  np.array([1,2,3,4,5,6])
multi_dim_arr = np.reshape(one_dim_arr,(2,3))

print(multi_dim_arr)

print(multi_dim_arr.ndim)

print(multi_dim_arr.shape)

print(multi_dim_arr.size)

#Array Math Operations

arr_1 = np.array([2,4,6])
arr_2 = np.array([1,3,5])

addition = arr_1 + arr_2
print(addition)

subtraction = arr_1 - arr_2
print(subtraction)

multiplication = arr_1 * arr_2
print(multiplication)

#Multiplying vector with scalar
vector = np.array([1,2])
print(vector * 1.6)

#Indexing and Slicing

a = np.array([1,2,3,4,5])
print(a[2])

print(a[0])

two_dim = np.array([[1,2,3],[4,5,6]])

print(two_dim[0][2])

print(two_dim[1][1])

sliced_arr = a[1:4]
print(sliced_arr)

sliced_arr = a[:3]
print(sliced_arr)

sliced_arr = a[2:]
print(sliced_arr)

sliced_arr = a[::2]
print(sliced_arr)

sliced_arr_1 = two_dim[0:2]
print(sliced_arr_1)

sliced_two_dim_rows = two_dim[1:3]
print(sliced_two_dim_rows)

sliced_two_dim_cols = two_dim[:,1]
print(sliced_two_dim_cols)


#Stacking

a1 = np.array([[1,1],[2,2]])
a2 = np.array([[3,3],[4,4]])

vert_stack = np.vstack((a1, a2))
print(vert_stack)

horz_stack = np.hstack((a1, a2))
print(horz_stack)