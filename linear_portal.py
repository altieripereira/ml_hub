import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

uri = "https://gist.githubusercontent.com/guilhermesilveira/b9dd8e4b62b9e22ebcb9c8e89c271de4/raw/c69ec4b708fba03c445397b6a361db4345c83d7a/tracking.csv"
dados = pd.read_csv(uri)

x = dados[['inicial', 'palestras', 'contato', 'patrocinio']]
y = dados["comprou"]

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.2, random_state=43)

teste_x.value_counts()
teste_y.value_counts()

model = LinearSVC()
model.fit(treino_x, treino_y)

previsoes = model.predict(teste_x)

accuracy = accuracy_score(teste_y, previsoes)

print(previsoes)
print(f"Acurácia: {accuracy * 100:.2f}%")
