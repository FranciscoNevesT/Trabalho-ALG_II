import numpy as np
from sklearn.model_selection import train_test_split
from kd import *
from dataset import *
import multiprocessing
from multiprocessing import Pool

class x_NN():
    """ Classe que representa um modelo de classificação kneighbors

        Attributes:
            self.X_train : np.array
              Contem os dados de treinamento

            self.y_train : np.array
              Contem a label dos dados de treinamento

            self.X_test : np.array
              Contem os dados de teste

            self.y_test : np.array
              Contem a label dos dados de teste

            self.kd_tree : dict
              Uma kd_tree com os dados de treinamento
        """

    def fit(self, path,test_size=0.3):
        """ Função que separa os conjuntos de treino e teste e cria a arvore kd

            Parameters:
                path : string
                  Local aonde está o dataset

                test_size : int, default = 0.3
                  Proporção do conjunto de teste em relação à todos od dados
        """
        data = Dataset(path = path).dataset[:200]

        self.X_train = data[:,:-1]
        self.y_train = data[:,-1]

        self.X_test = []
        self.y_test = []

        split = int(test_size * len(data))

        for _ in range(split):
            sep = np.random.randint(0,len(self.X_train))

            self.X_test.append(self.X_train[sep])
            self.y_test.append(self.y_train[sep])

            self.X_train = np.delete(self.X_train, sep , 0)
            self.y_train = np.delete(self.y_train, sep , 0)

        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        self.kd_tree = kd(self.X_train)

    def knn(self, k_size, cpu = -1):
        """ Função que encontra os k_size pontos mais proximos de todos os ponto em self.X_test.

            Parameters:
                k_size : int
                  Numero de pontos a serem preditos

                cpu : int, default = -1
                  Numero de cpus para serem usadas, Caso -1, usa todas

            Return:
                predicts : list
                  conjunto de todos os k_size pontos para cada ponto em self.X_text

        """
        self.k_size = k_size

        if cpu == -1:
            with Pool(multiprocessing.cpu_count()) as p:
                predicts = p.map(self.multi_knn, self.X_test)
        elif cpu == 1 or cpu == 0:
            predicts = []
            for i in self.X_test:
                predicts.append(self.multi_knn(point= i))
        else:
            with Pool(cpu) as p:
                predicts = p.map(self.multi_knn, self.X_test)

        return predicts

    def multi_knn(self,point):
        """ Função intermediaria para ser possivel utilizar o multiprocessing.
                Parameters:
                    point : np.array
                      Ponto para ser encontra os k_size pontos mais proximos

                Return:
                    kneighbor : list
                      k_size pontos mais proximos de point

                """
        kneighbor = []
        self.knn_aux(point = point, tree= self.kd_tree.kd_tree, k_size= self.k_size, kneighbor= kneighbor,
                     maior_distancia=[-1,-1], check= [0], dists=[])

        return kneighbor

    def knn_aux(self, tree, point, k_size, kneighbor, maior_distancia, check, dists):
        """ Função que localiza os k_size mais proximos de point

                Parameters:
                    tree : dic
                      Arvore em que sera realizada a busca
                    point : np.array
                      Ponto a ser localiza os k_size pontos mais proximos
                    k_size : int
                      Numero de neighbor a ser localidado
                    kneighbor : list
                      Pontos mais proximos
                    maior_distancia : int
                      Maior distanccia do ponto mais longe de point em kneighbor
                    check : list
                      Conta o numero de interações do algoritmo
                    dists : list
                      Lista com as distancias entre o kneighbor e point
                """
        if type(tree["POINT"]) == type(np.array([])):
            dist = np.linalg.norm(point - tree["POINT"])

            if len(kneighbor) < k_size:
                kneighbor.append(tree["POINT"])
                dists.append(dist)

                if dist > maior_distancia[0]:
                    maior_distancia[0] = dist
                    maior_distancia[1] = len(kneighbor) - 1
            else:
                if dist < maior_distancia[0]:
                    tirar = maior_distancia[1]

                    kneighbor.pop(tirar)
                    kneighbor.append(tree["POINT"])

                    dists[tirar] = dist
                    maior_distancia[0] = np.max(dists)
                    maior_distancia[1] = np.argmax(dists)

            check[0] = check[0] + 1

        else:
            corte = tree["CORTE"]
            dim = tree["DIM"]

            dif = abs(corte - point[dim])

            if len(kneighbor) < k_size or dif < maior_distancia[0]:
                menor = tree["MENOR"]
                self.knn_aux(tree=menor, point=point, k_size=k_size,
                             kneighbor=kneighbor, maior_distancia=maior_distancia, check=check, dists=dists)
                maior = tree["MAIOR"]
                self.knn_aux(tree=maior, point=point, k_size=k_size,
                             kneighbor=kneighbor, maior_distancia=maior_distancia, check=check, dists=dists)
            else:
                if point[dim] < corte:
                    menor = tree["MENOR"]
                    self.knn_aux(tree=menor, point=point, k_size=k_size,
                                 kneighbor=kneighbor, maior_distancia=maior_distancia, check=check, dists=dists)
                else:
                    maior = tree["MAIOR"]
                    self.knn_aux(tree=maior, point=point, k_size=k_size,
                                 kneighbor=kneighbor, maior_distancia=maior_distancia, check=check, dists=dists)

    def define_class(self, kneighbors):
        """ Função que classifica a classe considerando os kneighbors

                Parameters:
                    kneighbors : list
                      Lista contendo k_size neighbors para o ponto

                 Return:
                    classe_pred : int
                      Classe do ponto

                """
        k_classes = {}

        for n in kneighbors:
            i = 0

            for p in self.X_train:
                if (p == n).all():
                    break

                i += 1

            classe = self.y_train[i]
            if list(k_classes.keys()).count(classe):
                k_classes[classe] = k_classes[classe] + 1
            else:
                k_classes[classe] = 1

        maior =  np.max(list(k_classes.values()))

        classe_pred = []

        for i in k_classes.keys():
            if k_classes[i] == maior:
                classe_pred.append(i)

        return classe_pred[np.random.randint(len(classe_pred))]

    def predict(self, k_size, cpu):
        """ Função que prediz a classe de todos os valores em self.X_test

                Parameters:
                    k_size : int
                      Numero de pontos a serem preditos

                    cpu : int, default = -1
                      Numero de cpus para serem usadas, Caso -1, usa todas

                 Return:
                    classes_pred : List
                      Classe de todos os pontos

                """

        pred = self.knn(k_size=k_size, cpu=cpu)

        classes_pred = []

        for i in pred:
            classes_pred.append(self.define_class(kneighbors=i))

        return classes_pred

    def evaluate(self, k_size, cpu = -1):
        """ Função que calcula as metricas de acerto

                Parameters:
                    k_size : int
                      Numero de pontos a serem preditos

                    cpu : int, default = -1
                      Numero de cpus para serem usadas, Caso -1, usa todas

                 Return:
                    classes_pred : dict
                      Valores de acuracia, revocação e precição

                """
        metrics = {"acuracia": -1, "revocacao": -1, "precicao": -1}

        classes_pred = self.predict(k_size = k_size, cpu = cpu)


        classe = np.unique(self.y_test, return_counts= True)
        classe = classe[0][np.argmax(classe[1])]

        tp,tn,fp,fn = self.evaluate_aux(c = classe, pred= classes_pred)

        metrics["acuracia"] = (tp + tn) /(tp + tn + fp + fn)
        metrics["precicao"] = (tp) / (tp + fp)
        metrics["revocacao"] = (tp) / (tp + fn)

        return metrics

    def evaluate_aux(self, c, pred):
        """ Função que calcula as metricas de acerto

                Parameters:
                    c : int
                      Classe tomada como padrão

                    pred : list
                      Classes previstas para os pontos

                 Return:
                    [tp,tn,fp,fn] : list
                      Valores de true_positives(tp), true_negatives(tn), false_positives(fp) e false_negatives(fp)

                """
        obj = self.y_test

        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(obj)):
            y = obj[i]
            y_pred = pred[i]

            if y == c:
                if y == y_pred:
                    tp += 1
                else:
                    fp += 1
            else:
                if y == y_pred:
                    tn += 1
                else:
                    fn += 1

        return [tp,tn,fp,fn]