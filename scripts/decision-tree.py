import pandas as pd
import numpy as np
import seaborn as sns
import graphviz
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import export_graphviz
from datetime import datetime

dados = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/dd7ba8142321c2c8aaa0ddd6c8862fcc/raw/e694a9b43bae4d52b6c990a5654a193c3f870750/precos.csv")

dados['km_por_ano'] = dados['milhas_por_ano'] * 1.60934
dados['idade'] = datetime.today().year - dados['ano_do_modelo']
dados.drop(["milhas_por_ano", "ano_do_modelo"], axis=1, inplace=True)

dados.head()

x = dados[['preco', 'idade', 'km_por_ano']]
y = dados['vendido']

SEED = 42

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y,
                                                         random_state = SEED,
                                                         stratify = y)

print(f"Treinaremos com {len(raw_treino_x)}")
print(f"Testaremos com {len(raw_teste_x)}")

scaler = StandardScaler()
scaler.fit(raw_treino_x)

# treino_x = scaler.transform(raw_treino_x)
# teste_x = scaler.transform(raw_teste_x)

modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

classificador = DummyClassifier()
classificador.fit(treino_x, treino_y)
previsoes_baseline = classificador.predict(teste_x)
print(f"A acurácia do baseline foi de {accuracy_score(teste_y, previsoes_baseline) * 100:.2f}%")

acuracia = accuracy_score(teste_y, previsoes) * 100
print(f"A acurácia foi de {acuracia:.2f}%")

estrutura = export_graphviz(modelo, feature_names=x.columns, filled=True, rounded=True)
grafico = graphviz.Source(estrutura)
grafico

