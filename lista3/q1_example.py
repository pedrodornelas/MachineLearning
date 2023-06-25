import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Carregar o conjunto de dados a partir do arquivo CSV
data = pd.read_csv('data/data_bayesian_question.csv', header=None)

# Separar os dados rotulados com "+1" e "-1"
data_pos = data[data[2] == 1]
data_neg = data[data[2] == -1]

# Calcular os parâmetros estatísticos
u = np.mean(data_pos[[0, 1]], axis=0)  # Vetor de média para a classe "+1"
v = np.mean(data_neg[[0, 1]], axis=0)  # Vetor de média para a classe "-1"
Sigma = np.cov(data_pos[[0, 1]].T)  # Matriz de covariância para a classe "+1"
Omega = np.cov(data_neg[[0, 1]].T)  # Matriz de covariância para a classe "-1"

# Definir as probabilidades a priori
P_pos = 0.7
P_neg = 0.3

# Calcular os coeficientes e termos de viés para cada classe
W_pos = -0.5 * np.linalg.inv(Sigma)
W_neg = -0.5 * np.linalg.inv(Omega)
w_pos = np.dot(np.linalg.inv(Sigma), u)
w_neg = np.dot(np.linalg.inv(Omega), v)
w0_pos = -0.5 * np.dot(np.dot(u, np.linalg.inv(Sigma)), u) - 0.5 * np.log(np.linalg.det(Sigma)) + np.log(P_pos)
w0_neg = -0.5 * np.dot(np.dot(v, np.linalg.inv(Omega)), v) - 0.5 * np.log(np.linalg.det(Omega)) + np.log(P_neg)

# Funções discriminantes
def discriminant_pos(x):
    return np.dot(np.dot(x, W_pos), x) + np.dot(w_pos, x) + w0_pos

def discriminant_neg(x):
    return np.dot(np.dot(x, W_neg), x) + np.dot(w_neg, x) + w0_neg

# Classificação e cálculo da taxa de erro
error_count = 0
for _, sample in data.iterrows():
    x = sample[[0, 1]].values
    true_label = sample[2]
    if discriminant_pos(x) > discriminant_neg(x):
        predicted_label = 1
    else:
        predicted_label = -1
    if predicted_label != true_label:
        error_count += 1

error_rate = error_count / len(data)

# Plotar os dados e a fronteira de decisão
x1_range = np.linspace(-10, 10, 100)
x2_range = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = np.zeros_like(X1)

for i in range(len(x1_range)):
    for j in range(len(x2_range)):
        x = np.array([x1_range[i], x2_range[j]])
        if discriminant_pos(x) > discriminant_neg(x):
            Z[j, i] = 1
        else:
            Z[j, i] = -1

plt.figure(figsize=(8, 6))
plt.scatter(data_pos[0], data_pos[1], color='blue', label='+1')
plt.scatter(data_neg[0], data_neg[1], color='red', label='-1')
plt.contourf(X1, X2, Z, alpha=0.3, cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bayesian Classifier - Decision Boundary')
plt.legend()
plt.show()

print("Taxa de erro:", error_rate)
