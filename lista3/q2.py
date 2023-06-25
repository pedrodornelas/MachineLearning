import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import dataset
data = pd.read_csv("data/data_pca_question.csv", header=None)
print(data.head())

# calcular a média dos dados de cada feature
mean = np.mean(data, axis=0)
print("Mean:")
print(mean)

# para normalizar também devemos calcular os desvios padrões de cada features
std = np.std(data, axis=0)

data_norm = (data - mean)/std
print("Normalized data:")
print(data_norm.head())
# verificando se dados estão normalizados agora
# mean ~ 0, var = 1
# print("Mean: " + str(np.mean(data_norm, axis=0)))
# print("Var: " + str(np.var(data_norm, axis=0)))

# calcular a matriz de covariância dos dados
# é necessário fazer a matriz transposta dos dados normalizados 
# antes de fazer a covariancia
cov = np.cov(data_norm.T)
print("Cov: ")
print(cov)

autovalores, autovetores = np.linalg.eig(cov)
print("Autovalores: ")
print(autovalores)
print("Autovetores: ")
print(autovetores)

# para decidir sobre a reducao da dimensionalidade é necessário ver a representação
# da variabilidade através dos maiores autovalores
autovalores_ord = np.sort(autovalores)[::-1] # ordem descendente
# print(autovalores_ord)

total = np.sum(autovalores_ord)

# reducao para 1D
red = autovalores_ord[0] / total
print("1D: " + str(round(red*100,2)) + '% da variabilidade mantida')

# reducao para 2D
red = (autovalores_ord[0]+autovalores_ord[1]) / total
print("2D: " + str(round(red*100,2)) + '% da variabilidade mantida')

# reconstrucao dos dados nas componentes principais (autovetores) de maiores autovalores
autovetores_2d = autovetores[:, :2] # selecionar todas as linhas das 2 primeiras colunas
# print(autovetores_2d)

autovetores_1d = autovetores[:, :1] # selecionar todas as linhas da primeira coluna
# print(autovetores_1d)

# apos selecionar os autovetores das componentes principais podemos projetar os dados
data_2d = np.dot(data_norm, autovetores_2d)
data_1d = np.dot(data_norm, autovetores_1d)

# reconstrucao dos dados
data_recon_2d = np.dot(data_2d, autovetores_2d.T)
print(data_norm.head())   # apenas comparando os dados
print(data_recon_2d[:5])
data_recon_1d = np.dot(data_1d, autovetores_1d.T)
print(data_recon_1d[:5])

# plotar os dados originais, a projeção 1D e a projeção 2D em um mesmo gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Dados originais
ax.scatter(data_norm.iloc[:, 0], data_norm.iloc[:, 1], data_norm.iloc[:, 2], c='b', s=10, label='Original Data')

# Dados reconstruídos a partir da projeção 1D
ax.scatter(data_recon_1d[:, 0], data_recon_1d[:, 1], data_recon_1d[:, 2], c='r', s=10, label='Reconstructed 1D')

# Dados reconstruídos a partir da projeção 2D
ax.scatter(data_recon_2d[:, 0], data_recon_2d[:, 1], data_recon_2d[:, 2], c='g', s=10, label='Reconstructed 2D')

ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.legend()
plt.show()