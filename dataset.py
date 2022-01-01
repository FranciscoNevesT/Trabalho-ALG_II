import pandas as pd

class Dataset():
    def __init__(self, path):
        with open(path, "r") as file:
            self.text = file.read().split("@")[1:]

        self.data = {}

        for i in self.text:
            description = i.split()[0]
            value = i.split()[1:]

            if description == "attribute":
                description = value[0]
                value = value[1:]

            self.data[description] = value

        self.create_dataset()

    def create_dataset(self):

        values = []

        for i in self.data["data"]:
            value = i.split(",")

            value = [float(ii) for ii in value]

            values.append(value)

        numbers_i = [i for i in range(len(self.data["inputs"]))]

        self.dataset = pd.DataFrame(values, columns=numbers_i + ["out"])

        self.dataset["out"], _ = pd.factorize(self.dataset["out"])

        self.input = self.dataset.drop("out", axis=1)
        self.out = self.dataset[["out"]]

print(Dataset(path="dataset/banana.dat").dataset)