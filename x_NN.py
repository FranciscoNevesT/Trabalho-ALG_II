import numpy as np
from sklearn.model_selection import train_test_split
from kd import *
from dataset import *
import multiprocessing
from multiprocessing import Pool

class x_NN():
    def fit(self, path,test_size=0.3):
        data = Dataset(path = path).dataset[:100]

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

    def knn_aux(self, tree, point, k_size, kneighbor, maior_distancia, check, dists):
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

        e = []

        for i in k_classes.keys():
            if k_classes[i] == maior:
                e.append(i)

        return e[np.random.randint(len(e))]

    def multi_knn(self,point):
        kneighbor = []
        self.knn_aux(point = point, tree= self.kd_tree.kd_tree, k_size= self.k_size, kneighbor= kneighbor,
                     maior_distancia=[-1,-1], check= [0], dists=[])

        return kneighbor

    def knn(self, k_size, cpu = -1):
        self.k_size = k_size

        if cpu == -1:
            with Pool(multiprocessing.cpu_count()) as p:
                predicts = p.map(self.multi_knn, self.X_test)
        elif cpu == 1:
            predicts = []
            for i in self.X_test:
                predicts.append(self.multi_knn(point= i))

        return predicts

    def evaluate_aux(self, c, pred):
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


    def predict(self, k_size, cpu):
        pred = self.knn(k_size=k_size, cpu=cpu)

        classes_pred = []

        for i in pred:
            classes_pred.append(self.define_class(kneighbors=i))

        return classes_pred

    def evaluate(self, k_size, cpu = -1):
        metrics = {"acuracia": -1, "revocacao": -1, "precicao": -1}

        classes_pred = self.predict(k_size = k_size, cpu = cpu)


        classe = np.unique(self.y_test, return_counts= True)
        classe = classe[0][np.argmax(classe[1])]

        tp,tn,fp,fn = self.evaluate_aux(c = classe, pred= classes_pred)

        metrics["acuracia"] = (tp + tn) /(tp + tn + fp + fn)
        metrics["precicao"] = (tp) / (tp + fp)
        metrics["revocacao"] = (tp) / (tp + fn)

        return metrics