import pandas as pd

# read dataset
data_frame = pd.read_csv('Data_Iris.csv',sep=',')
print(data_frame.head())

# feature1 and feature2 means
mean1 = data_frame['Feature1'].mean()
mean2 = data_frame['Feature2'].mean()

print('Média 1: '+str(round(mean1,4)))
print('Média 2: '+str(round(mean2,4)))

# variance
var1 = data_frame['Feature1'].var()
var2 = data_frame['Feature2'].var()

print('Variância 1: '+str(round(var1,4)))
print('Variância 2: '+str(round(var2,4)))

# standart deviation -> sqrt(var)
std1 = var1**(1/2)
std2 = var2**(1/2)

print('Desvio Padrão 1: '+str(round(std1,4)))
print('Desvio Padrão 2: '+str(round(std2,4)))

# Covariance
sum = 0
n = len(data_frame['Feature1'])
for i in range(n):
    x1 = data_frame['Feature1'][i]
    x2 = data_frame['Feature1'][i]
    sum += (x1-mean1)*(x2-mean2)

Cov = round(sum/n,4)
print('Cov(x1,x2) = '+str(Cov))

Corr = round(Cov/(std1*std2),4)
print('rij = '+str(Corr))