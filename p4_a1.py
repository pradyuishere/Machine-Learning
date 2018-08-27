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
#Calculating the weights matrix and squared error
weights_matrix_calc= (np.linalg.inv(x_feature.T*x_feature))*x_feature.T *y
predicted_outputs=x_feature*weights_matrix_calc
varience=(sum(np.power((y-predicted_outputs), 2)))/N
#print(np.sqrt(varience))
#print(np.random.normal(0, np.sqrt(varience), [N,1]))
actual_outputs_with_noise=predicted_outputs

plt.plot(x.T, y, 'ro', x.T, actual_outputs_with_noise, 'g.')
plt.show()
print("**************This problem is just a repeat of problem 2 or problem 1 if degree is set to 1.**************")
