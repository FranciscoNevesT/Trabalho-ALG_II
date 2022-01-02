from sklearn.model_selection import train_test_split
from kd import *
from dataset import *
import time
import multiprocessing
from multiprocessing import Pool

class x_NN():
    def fit(self, path,test_size=0.3):
        data = Dataset(path = path).dataset[:100]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data.drop("out", 1),
                                                                                data["out"], test_size=test_size)

        self.kd_tree = kd(self.X_train)

    def knn_aux(self, tree, point, k_size, kneighbor, maior_distancia, check, dists):
        if tree == None:
            1 + 1
        elif type(tree["POINT"]) == type(np.array([])):
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
        e = []
        for n in kneighbors:
            p = self.X_train
            for dim in range(len(n)):
                p = p[p[dim] == n[dim]]

            e.append(self.y_train[p.index[0]])

        contar = pd.DataFrame(e)[0].value_counts()
        contar = contar[contar == contar.iloc[0]]

        maiores = contar.index

        return maiores[np.random.randint(len(maiores))]

    def multi_knn(self,point):
        kneighbor = []
        self.knn_aux(point = point, tree= self.kd_tree.kd_tree, k_size= self.k_size, kneighbor= kneighbor,
                     maior_distancia=[-1,-1], check= [0], dists=[])

        return kneighbor

    def knn(self, k_size, cpu = -1):
        self.k_size = k_size

        if cpu == -1:
            with Pool(multiprocessing.cpu_count()) as p:
                if __name__ == '__main__':
                    predicts = p.map(self.multi_knn, self.X_test.values)
        elif cpu == 1:
            predicts = []
            for i in self.X_test.values:
                predicts.append(self.multi_knn(point= i))

        return predicts

    def evaluate_aux(self, c, pred):
        obj = self.y_test.values

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


    def evaluate(self, k_size, cpu = -1):
        metrics = {"acuracia": -1, "revocacao": -1, "precicao": -1}

        tp = 0
        tn = 0
        fp = 0
        fn = 0

        pred = self.knn(k_size=k_size, cpu=cpu)

        classes_pred = []

        for i in pred:
            classes_pred.append(self.define_class(kneighbors=i))

        classes = self.y_test.unique()

        for c in classes:
            tp_i,tn_i,fp_i,fn_i = self.evaluate_aux(c = c, pred= classes_pred)

            tp += tp_i
            tn += tn_i
            fp += fp_i
            fn += fn_i

        metrics["acuracia"] = (tp + tn) /(tp + tn + fp + fn)
        metrics["precicao"] = (tp) / (tp + fp)
        metrics["revocacao"] = (tp) / (tp + fn)

        return metrics