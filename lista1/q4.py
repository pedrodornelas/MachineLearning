import pandas as pd

# mean
def mean(x):
    n = len(x)
    sum = 0
    for xi in x:
        sum += xi
    return sum/n

# variance
def var(x):
    n = len(x)
    me = mean(x)
    sum = 0
    for xi in x:
        sum += (xi-me)**2
    return sum/n

# standart deviation
def std(x):
    return var(x)**(1/2)

# covariance
def cov(x,y):
    if len(x) == len(y):
        n = len(x)
        me1 = mean(x)
        me2 = mean(y)
        sum = 0
        for i in range(n):
            sum += (x[i]-me1)*(y[i]-me2)
        return sum/n
    else:
        print('Error')
        exit(-1)

def corr(x,y):
    std1 = std(x)
    std2 = std(y)
    Cov = cov(x,y)
    return Cov/(std1*std2)

# read dataset
data_frame = pd.read_csv('Data_Iris.csv', sep=',')
print(data_frame.head())

x = data_frame['Feature1']
y = data_frame['Feature2']

# feature1 and feature2 means
mean1 = mean(x)
mean2 = mean(y)

print('Média 1: '+str(round(mean1, 4)))
print('Média 2: '+str(round(mean2, 4)))

# variance
var1 = var(x)
var2 = var(y)

print('Variância 1: '+str(round(var1, 4)))
print('Variância 2: '+str(round(var2, 4)))

# standart deviation -> sqrt(var)
std1 = std(x)
std2 = std(y)

print('Desvio Padrão 1: '+str(round(std1,4)))
print('Desvio Padrão 2: '+str(round(std2,4)))

# covariance
Cov = cov(x,y)
print('Cov(x,y)||sij = '+str(round(Cov,4)))

# correlation coefficient
Corr = corr(x,y)
# Corr = round(Cov/(std1*std2),4)
print('rij = '+str(round(Corr,4)))