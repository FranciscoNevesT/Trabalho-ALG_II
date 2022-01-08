import matplotlib.pyplot as plt
import os
import numpy
from x_NN import *

i = 3
j = 4

fig, ax = plt.subplots(i,j)

plt.rcParams['figure.figsize']  = (16, 10)
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['axes.titlesize']  = 10
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['lines.linewidth'] = 4
plt.ylim([0, 1.2])

fig.suptitle("Avaliação dos datasets")

plt.style.use('seaborn-colorblind')

k_size = [1,2,4,6,8,10]

model = x_NN()
a = 0
b = 0



acuracia_global = {}
revocacao_global = {}
precicao_global = {}

for k in k_size:
    acuracia_global[k] = []
    revocacao_global[k] = []
    precicao_global[k] = []

for file in os.listdir("dataset"):
    model.fit(path= "dataset/" + file)

    acuracia = []
    revocacao = []
    precicao = []

    for k in k_size:
        if __name__ == '__main__':
            dados = model.evaluate(k_size=k,cpu = 1)

            acuracia.append([dados["acuracia"]])
            acuracia_global[k].append(dados["acuracia"])

            revocacao.append([dados["revocacao"]])
            revocacao_global[k].append(dados["revocacao"])

            precicao.append([dados["precicao"]])
            precicao_global[k].append(dados["precicao"])

    ax[a][b].set_title(file)
    ax[a][b].plot(k_size,acuracia, label = "acuracia")
    ax[a][b].plot(k_size,revocacao, label = "revocacao")
    ax[a][b].plot(k_size,precicao, label = "precicao")
    ax[a][b].set_ylim([0,1.1])

    a += 1
    if a == i:
        a = 0
        b += 1

plt.legend()
plt.show()

for k in k_size:
    acuracia_global[k] = np.mean(acuracia_global[k])
    revocacao_global[k] = np.mean(revocacao_global[k])
    precicao_global[k] = np.mean(precicao_global[k])

plt.title("Media das ?")
plt.plot(acuracia_global.keys(),acuracia_global.values(), label = "acuracia")
plt.plot(revocacao_global.keys(),revocacao_global.values(), label = "revocacao")
plt.plot(precicao_global.keys(),precicao_global.values(), label = "precicao")
plt.ylim([0,1.1])
plt.legend()
plt.show()

print(acuracia_global)