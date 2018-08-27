import csv
import numpy as np
#_______________________________________________________________________________
datafile = open('X.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
datax = []
for row in datareader:
    datax.append(row)
datax=np.array(datax).astype(float)
datax=datax.T
print(type(datax[0][0]))
#_______________________________________________________________________________
datafile = open('Y.csv', 'r')
datareader = csv.reader(datafile, delimiter=',')
datay = []
for row in datareader:
    datay.append(row)
datay=np.array(datay).astype(float)
#_______________________________________________________________________________
means=[]
vars=[]
#print(np.matrix([datax[j, :] for j in range(np.size(datay,0)) if float(datay[j])==1.0]))

for i in [-1, 1]:
   means.append(np.mean(np.matrix([datax[j, :].tolist() for j in range(np.size(datay,0)) if float(datay[j])==i]).astype(float),axis=0).tolist())
   vars.append(np.std(np.matrix([datax[j, :].tolist() for j in range(np.size(datay,0)) if float(datay[j])==i]).astype(float),axis=0).tolist())

#_______________________________________________________________________________
#Predictions
inputs=[]
for iter in range(np.size(datax, 1)):
    inputs.append(float(input("input x:")))


probability0=1
probability1=1

for iter in range(np.size(datax, 1)):
    probability0=probability0*np.exp(-np.power(inputs[:][iter]-means[0][0][iter],2)/(2*vars[0][0][iter]*vars[0][0][iter]))/np.sqrt(2*np.pi*vars[0][0][iter])
    probability1=probability1*np.exp(-np.power(inputs[:][iter]-means[1][0][iter],2)/(2*vars[1][0][iter]*vars[1][0][iter]))/np.sqrt(2*np.pi*vars[1][0][iter])

count0=0
count1=0
for iter in range(np.size(datax,0)):
    if datay[iter]==-1:
        count0=count0+1
    else:
        count1=count1+1
print(count0)
probability0=probability0*count0*1.0/np.size(datay,0)
probability1=probability1*count1/np.size(datay,0)


print("probability0 :",probability0)
print("probability1 :", probability1)

if probability0>probability1:
    print("predicted y : 0")
else:
    print("predicted y : 1")
