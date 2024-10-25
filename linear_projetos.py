import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

uri = "https://gist.githubusercontent.com/guilhermesilveira/12291c548acaf544596795709020e3db/raw/325bdef098bd9cbc2189215b7e32e22f437f29f3/projetos.csv"

dados = pd.read_csv(uri)

dados['finalizado'] = dados['nao_finalizado'].map({1: 0, 0: 1})
dados = dados.drop(columns=['nao_finalizado'])
dados = dados.query('horas_esperadas > 0')
dados.head()

x = dados[['horas_esperadas', 'preco']]
y = dados['finalizado']

raw_teste_x, raw_treino_x, teste_y, treino_y = train_test_split(x, y, test_size=0.2, random_state=2236)

scaler = StandardScaler()
scaler.fit(raw_treino_x)
treino_x = scaler.transform(raw_treino_x)
teste_x = scaler.transform(raw_teste_x)

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)
accuracy = accuracy_score(teste_y, previsoes)

print(f"Acurácia: {accuracy * 100:.2f}%")

# sns.scatterplot(x='horas_esperadas', y='preco', data=dados, hue='finalizado')
# plt.show()

# sns.relplot(x='horas_esperadas', y='preco', data=dados, hue='finalizado', col='finalizado')
# plt.show()

# x_min = teste_x['horas_esperadas'].min()
# x_max = teste_x['horas_esperadas'].max()
# y_min = teste_x['preco'].min()
# y_max = teste_x['preco'].max()

# eixo_x = np.arange(x_min, x_max, (x_max - x_min) / 100)
# eixo_y = np.arange(y_min, y_max, (y_max - y_min) / 100)

# xx, yy = np.meshgrid(eixo_x, eixo_y)
# pontos = np.c_[xx.ravel(), yy.ravel()]

# Z = modelo.predict(pontos)
# Z = Z.reshape(xx.shape)


# plt.contourf(xx, yy, Z, alpha=0.3)
# sns.scatterplot(x='horas_esperadas', y='preco', data=teste_x, hue=teste_y)
# plt.show()
