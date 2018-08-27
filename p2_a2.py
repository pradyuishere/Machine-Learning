import csv
import numpy as np
import copy
#_______________________________________________________________________________
def bubble(a,b):
    c=copy.deepcopy(a)
    print(type(c))
    while(1):
        count=0
        for i in range(np.size(c,0)-1):
            if c[i,b]>c[i+1,b]:
                temp=copy.deepcopy(c[i,:])
                c[i,:]=copy.deepcopy(c[i+1,:])
                c[i+1,:]=copy.deepcopy(temp)
                count=count+1
        if count==0:
            #print(np.concatenate((a,c), axis=1))
            return np.matrix(c)

#_______________________________________________________________________________
datafile = open('X.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
datax = []
for row in datareader:
    datax.append(row)

datax=np.array((datax)).astype(float)
datax=datax.T

#_______________________________________________________________________________
datafile = open('Y.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
datay = []
for row in datareader:
    datay.append(row)
datay=np.array(datay).astype(float)
#_______________________________________________________________________________
k_number=int(input("enter the value of 'k' : "))

inputs=[]
for iter in range(np.size(datax, 1)):
    inputs.append(float(input("input x:")))

distances=0

for iter in range(np.size(datax,1)):
    #print(iter)
    distances=distances+np.power(inputs[iter]-datax[:,iter],2)

#print(np.shape(np.matrix(distances)))
distances=np.concatenate((np.matrix(distances).T,datay),axis=1)
#distances=np.concatenate((np.matrix(np.sqrt(np.power(x1-datax[:,0],2)+np.power(x2-datax[:,1],2))).T, datay),axis=1)
distances_1=(bubble(distances,0))

print(type(distances_1))
print(np.shape(distances_1))
print(distances_1[:k_number,:])

means=np.mean(distances_1[0:k_number, 1].astype(float))

if means >=0:
    print("y_prediction : 1")
else:
    print("y_prediction : -1")
