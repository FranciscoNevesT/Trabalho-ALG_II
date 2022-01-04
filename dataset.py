import pandas as pd

class Dataset():
    """ Classe para automaticamente formatar os arquivos

    Parameters:
        path : string
          Caminho para o arquivo .dat

    Attributes:
        sumario : dict
          Todas as informações contidas no arquivo
        dataset : pd.Dataframe
          Valores das instacias com out sendo a classe
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

        for i in text:
            description = i.split()[0]
            value = i.split()[1:]

            if description == "attribute":
                description = value[0]
                value = value[1:]

            self.sumario[description] = value

        self._create_dataset()

    def _create_dataset(self):
        """
        Cria um pd.Dataframe dos pontos
        """

        #Pegando os pontos
        pontos = []

        for p in self.sumario["data"]:
            ponto = p.split(",")

            ponto = [float(i) for i in ponto]

            pontos.append(ponto)

        #Criando o dataset
        dimensions = [i for i in range(len(self.sumario["inputs"]))]

        dimensions.append("out")

        self.dataset = pd.DataFrame(pontos, columns=dimensions)



        #Convertendo os valores das classes para numerico
        self.dataset["out"], classes = pd.factorize(self.dataset["out"])
        self.dataset["out"] = [int(i) for i in self.dataset["out"].values]

        #fazendo o indexador
        self.indexador = {}

        for i in range(len(classes)):
            self.indexador[i] = classes[i]
