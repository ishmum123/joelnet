"""
The canonical example of a function that can't be
learned with a simple linear model is XOR
"""
import numpy as np

from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh

from iris_dataset import dataset

def extract_data(data): return data[:3]

def extract_category(data): return data[4:][0]

def category_to_assigned_array(category):
    if category == "Iris-setosa": return [0, 0, 1]
    elif category == "Iris-versicolor": return [0, 1, 0]
    elif category == "Iris-virginica": return [1, 0, 0]
    else: return "Undefined"

def assigned_array_to_category(assigned_array):
    if assigned_array == [0, 0, 1]: return "Iris-setosa"
    elif assigned_array == [0, 1, 0]: return "Iris-versicolor"
    elif assigned_array == [1, 0, 0]: return "Iris-virginica"
    else: return "Undefined"

def get_input(): return input("Epochs OR q\n")

input_list = list(map(extract_data, dataset))

inputs = np.array(input_list)

targets = np.array(list(map(category_to_assigned_array, list(map(extract_category, dataset)))))

key = get_input()

while key != "q":

    net = NeuralNet([
        Linear(input_size=3, output_size=3),
        Tanh(),
        Tanh(),
        Linear(input_size=3, output_size=3)
    ])

    train(net, inputs, targets, int(key))

    for x, y in zip(inputs, targets):
        predicted = net.forward(x)
        print(list(map(int, map(round, predicted.tolist()))))

    key = get_input()
