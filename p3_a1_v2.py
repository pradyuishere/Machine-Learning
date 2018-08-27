import numpy as np
import matplotlib.pyplot as plt
#_______________________________________________________________________________
#Generate the training samples
# Number of training samples
N = 10
# Generate equispaced floats in the interval [0, 2π]
x = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = float(input("Enter the required std of noise to be added to Y : "))
# Generate some numbers from the sine function
y = np.sin(x)
# Add noise
y += np.random.normal(mean, std, N)
##______________________________________________________________________________
#Creating the feature matrix
#Enter the degree of the polynomial to be used
degree = int(input("Enter the polynomial order (not including the constant) :"))
lagrangian_multiplier=float(input("Enter a lagrangian_multiplier :"))

#Createing the feature matrix for the input
x_feature=np.ones(np.size(x))
x_feature=np.matrix(x_feature)
y=np.matrix(y)
#Appending each of the polynomial features
for i in range(degree):
    x_feature=np.concatenate((x_feature, np.matrix(np.power(x, (i+1)))), axis=0)
#_______________________________________________________________________________
#Now, all the Matrices formed so far need to be transposed as we formed as
#row matrixes at first and not as column matrices
x_feature=x_feature.T
y=y.T
#_______________________________________________________________________________
# Calculating the Lagrangian_identity matrix. Since the using the identity matrix
# means bias weights as well, we zero down the first entry of the Lagrangian_identity
# matrix.
iden_lag=np.eye(degree+1)
iden_lag[0][0]=0
#_______________________________________________________________________________
#Calculating the weights matrix and squared error
weights_matrix_calc= (np.linalg.inv(x_feature.T*x_feature+1.0*lagrangian_multiplier*iden_lag))*x_feature.T *y
predicted_outputs=x_feature*weights_matrix_calc
squared_error=sum(np.power((y-predicted_outputs), 2))
#print (weights_matrix_calc)
print("squared_error test : ", squared_error)
#_______________________________________________________________________________
#predictions against new outputs
x_new = np.matrix(np.linspace(0, 2*np.pi, 1000))
z=np.sin(x_new)

x_feature_new=np.ones(np.size(x_new))
x_feature_new=np.matrix(x_feature_new)
y=np.matrix(y)
#Appending each of the polynomial features
for i in range(degree):
    x_feature_new=np.concatenate((x_feature_new, np.matrix(np.power(x_new, (i+1)))), axis=0)

x_feature_new=x_feature_new.T
predicted_outputs_new=x_feature_new*weights_matrix_calc

plt.plot(x.T, y, 'ro', x_new.T, predicted_outputs_new, 'g,', x_new, z, 'k,')
plt.show()
