import pandas as pd
import numpy as np

class Dataset():
    """ Classe para automaticamente formatar os arquivos

    Parameters:
        path : string
          Caminho para o arquivo .dat

    Attributes:
        sumario : dict
          Todos os pontos do dataset
        dataset : np.array
          Valores das instacias com o ultimo valor sendo a classe
        indexador : dict
          Indexa os numeros com as classes originais
    """
    def __init__(self, path : str):
        """
        Abre o arquivo .dat e cria o sumario das suas informações

        :param path: caminho para o arquivo .dat
        """

        with open(path, "r") as file:
            text = file.read().split("@")[1:]

        self.sumario = {}

        data_v = False
        for i in text:
            description = i.split("\n")[0]
            value = i.split("\n")[1:]

            if description != "data":
                continue

            self.sumario[description] = value

        self._create_dataset()

    def _create_dataset(self):
        """
        Cria um np.array dos pontos
        """

        #Pegando os pontos
        points = []

        for p in self.sumario["data"]:
            point = p.split(",")
            point = " ".join(point)
            point = point.split()

            point_v = []

            for i in point:
                try:
                    point_v.append(float(i))
                except:
                    point_v.append(i)

            points.append(point_v)

        #Criando o dataset
        columns = [i for i in range(len(points[0]) - 1)]
        columns.append("out")
        self.dataset = pd.DataFrame(points, columns= columns)
        self.dataset.dropna(inplace=True)

        #Convertendo os valores das classes para numerico
        self.dataset["out"], classes = pd.factorize(self.dataset["out"])
        self.dataset["out"] = [int(i) for i in self.dataset["out"].values]

        #fazendo o indexador
        self.indexador = {}

        for i in range(len(classes)):
            self.indexador[i] = classes[i]

        self.dataset = self.dataset.values



