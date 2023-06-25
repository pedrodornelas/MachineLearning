import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# import dataset
# o dataset escolhido foi o Iris pela fácil interpretação dos dados e é comumente utilizado para testes
data = pd.read_csv('data/Data_Iris.csv')
# print(data.head())
features = data.iloc[:, :4] # Seleciona todas as features do dataset
classes = data['Class']     # Seleciona somente a classificação de cada amostra
# print(features.head())
# print(classes.head())

# verificar as 3 features com menor correlação entre as 4 features para aplicar o k-means
# monta a matrix de correlação entre as features
correlation_matrix = features.corr()    
# procura as 3 menores correlações em valores absolutos, ou seja, quanto mais próximo de 0, menos a correlação
low_corr_features = correlation_matrix.abs().unstack().nsmallest(3)
# seleciona as features com menor correlação para então aplicar o k-means
selected_features = low_corr_features.index.get_level_values(1).unique().tolist()
print("Features com menor correlação absoluta: " + str(selected_features))

data_input = data[selected_features]
N = len(features) # numero de amostras

best_k = 0
best_inercia = float('inf')
best_labels = []
best_centers = []
inercia_arr = []
flag = False

# tolerancia de erro de reconstrução
tol = 0.07

for k in range(1,15):
    # aplicando do algoritmo k-means
    # Obs, é necessário iniciar o numero k de clusters, por padrao sao 8
    kmeans = KMeans(n_clusters=k) 
    kmeans.fit(data_input)

    # obtendo os rotulos que foram atribuídos a cada amostra 
    labels = kmeans.labels_
    # obtendo as coordenadas dos centros dos clusters
    cluster_centers = kmeans.cluster_centers_

    # Somas das distancias quadradas entre os pontos e o centro do cluster mais proximo a amostra
    # podemos toma-la para decidir a quantidade de clusters ideal
    # semelhante ao erro de reconstrução
    inercia_arr.append(kmeans.inertia_/N)
    if best_inercia-(kmeans.inertia_/N) >= tol and flag == False:
        best_inercia = kmeans.inertia_/N
        best_k = k
        best_labels = labels
        best_centers = cluster_centers
    else:
        # condicao de parada em que o ponto do cotovelo foi encontrado
        flag = True

# ajustar as coordenadas dos centros
scaler = MinMaxScaler()
scaled_centers = scaler.fit_transform(best_centers)

# para visualizar em um gráfico 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[selected_features[0]], data[selected_features[1]], data[selected_features[2]], c=best_labels)

# obter as coordenadas dos centros dos clusters
x_centers = best_centers[:, 0]
y_centers = best_centers[:, 1]
z_centers = best_centers[:, 2]

# plotar os centros dos clusters
ax.scatter(x_centers, y_centers, z_centers, marker='x', color='red', s=100, linewidths=2)

ax.set_xlabel(selected_features[0])
ax.set_ylabel(selected_features[1])
ax.set_zlabel(selected_features[2])

# plot erro de reconstrução
fig2 = plt.figure()
x = np.arange(1,15)
plt.scatter(best_k, best_inercia, marker='x', color='red', label='Best k')
plt.plot(x, inercia_arr)
plt.xlabel("Number of Clusters")
plt.ylabel("Reconstruction Error")
plt.legend()

plt.show()