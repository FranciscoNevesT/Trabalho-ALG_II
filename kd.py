import numpy as np
import matplotlib.pyplot as plt
from dataset import *

class kd():
    """ Classe que representa a arovre kd

        Parameters:
            data : pd.DataFrame
              Dataset criado por dataset.py


        Attributes:
            sumario : dict
        """

    def __init__(self, data):
        self.kd_tree = self.make_kd(data)

    def create_node(self):
        """ Representação de um nó na arvore
              CORTE : O valor que divide a arvore
              DIM : Dimensão do valor do corte
              POINT : Guarda um dos pontos, caso seja um folha. Se não , seu valor é None
              MENOR : Arvore da esquerda com os valores menores que corte
              MAIOR : Arvore da direita com os valores maiores ou iguais que corte
        """

        node = {"CORTE": None,
                "DIM": -1,
                "POINT": [],
                "MENOR": None,
                "MAIOR": None}

        return node

    def insert_kd(self, tree, point, n_dim):
        """ Função que insere um ponto na arvore

            Parameters:
                tree : dict
                  Arvore KD
                point : np.array
                  Ponto a ser inserido
                n_dim : int
                  O numero de dimensões de Ponto
        """

        dim = tree["DIM"]

        if tree["CORTE"] == None:
            l_tree = self.create_node()
            l_tree["DIM"] = (dim + 1) % n_dim
            r_tree = self.create_node()
            r_tree["DIM"] = (dim + 1) % n_dim

            p0 = tree["POINT"]
            tree["POINT"] = None

            if p0[dim] < point[dim]:
                l_tree["POINT"] = p0
                r_tree["POINT"] = point
            else:
                l_tree["POINT"] = point
                r_tree["POINT"] = p0

            median = np.median([p0[dim], point[[dim]]])[0]

            tree["CORTE"] = median
            tree["MENOR"] = l_tree
            tree["MAIOR"] = r_tree
        else:
            if point[dim] < tree["CORTE"]:
                self.insert_kd(tree["MENOR"], point, n_dim)
            else:
                self.insert_kd(tree["MAIOR"], point, n_dim)

    def make_kd(self, data):
        """ Função que cria a arvore

            Parameters:
                data : np.array
                  Dataset criado por dataset.py
        """

        kd_tree = self.create_node()
        kd_tree["POINT"] = data[0]
        kd_tree["DIM"] = 0

        for point in data[1:]:
            self.insert_kd(kd_tree, point, n_dim=len(point))

        return kd_tree

    def get_deep(self):
        """
        Junto com _get_deep_aux, retorna a profundidade da arvore
        """
        return self._get_deep_aux(self.kd_tree)

    def _get_deep_aux(self, tree):
        if tree == None:
            return 0
        elif type(tree["POINT"]) == type(np.array([])):
            return 1
        else:
            return 1 + max(self._get_deep_aux(tree["MENOR"]), self._get_deep_aux(tree["MAIOR"]))

    def lines(self):
        """
        Junto com _lines_aux, desenha todas as linhas de separação
        """
        plt.figure(figsize=(10, 10))
        self._lines_aux(self.kd_tree)

    def _lines_aux(self, tree, bounds=[[-100, 100], [-100, 100]]):
        corte = tree["CORTE"]
        dim = tree["DIM"]

        if type(tree["POINT"]) == type(None):
            if dim == 0:
                a = bounds[1][0]
                b = bounds[1][1]

                plt.plot([corte, corte], [a, b], color="r")
                lbound = [[bounds[0][0], corte], [bounds[1][0], bounds[1][1]]]
                rbound = [[corte, bounds[0][1]], [bounds[1][0], bounds[1][1]]]

            else:
                a = bounds[0][0]
                b = bounds[0][1]

                plt.plot([a, b], [corte, corte], color="r")
                lbound = [[bounds[0][0], bounds[0][1]], [bounds[1][0], corte]]
                rbound = [[bounds[0][0], bounds[0][1]], [corte, bounds[1][1]]]

            if type(tree["MENOR"]) != type(None):
                self._lines_aux(tree=tree["MENOR"], bounds=lbound)

            if type(tree["MAIOR"]) != type(None):
                self._lines_aux(tree=tree["MAIOR"], bounds=rbound)