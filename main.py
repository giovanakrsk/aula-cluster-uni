import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataframe = pd.read_csv('dataset_alunos.csv')

colunas =["idade","genero","trabalha","pais","frequencia"]

valores_x = dataframe[colunas]

inercia = []
rangex = range(1,11)

for index in rangex:
    means = KMeans(n_clusters=index,random_state=42).fit(valores_x)
    inercia.append(means.inertia_)

plt.plot(rangex, inercia)
plt.xlabel("número de clusters")
plt.ylabel("inércia")
plt.title("método cotovelo - definição do N clusters")
plt.show()