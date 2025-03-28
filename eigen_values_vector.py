import tensorflow as tf


print("Matrix multiplication Demo")


x=tf.constant([1,2,3,4,5,6],shape=[2,3])
print(x)
y=tf.constant([7,8,9,10,11,12],shape=[3,2])
print(y)
z=tf.matmul(x,y)
print("Product",z)
mat_A=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,name="MatrixA")
print("Matrix A:\n{}\n\n".format(mat_A))
eigen_values_A,eigen_vectors_A=tf.linalg.eigh(mat_A)
print("Eigen Vectors:\n{}\n\nEigen Values:\n{}\n".format(eigen_vectors_A,eigen_values_A))
