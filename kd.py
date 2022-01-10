import numpy as np

class KD:
    """ Classe que representa a arvore kd

        Parameters:
            data : np.array
              Conjunto de pontos criado pela classe Dataset


        Attributes:
            self.kd_tree : dict
              Arvore kd criada usando os pontos de data
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
                  Arvore em que point tem que ser inserida
                point : np.array
                  Ponto a ser inserido
                n_dim : int
                  O numero de dimensões de point
        """

        dim = tree["DIM"]

        if isinstance(tree["CORTE"], type(None)):  # Caso a arvore seja uma folha
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

            median = np.median([p0[dim], point[dim]])

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

        # Inserindo todos os pontos
        kd_tree = self.create_node()
        kd_tree["POINT"] = data[0]
        kd_tree["DIM"] = 0

        for point in data[1:]:
            self.insert_kd(kd_tree, point, n_dim=len(point))

        return kd_tree
