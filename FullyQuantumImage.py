
import torch
import torch.utils.data
from torch import nn
import torchquantum as tq
import tqdm

from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(0)

BATCH_SIZE = 100
NUM_LAYER = 2
NUM_QUBITS = 4
EPOCHS = 10
LEARNING_RATE = 0.005

train_data = datasets.MNIST(
    root = './data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = './data', 
    train = False, 
    transform = ToTensor()
)


batch_size = BATCH_SIZE
train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)


num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
print("Layer =",num_layer)
print("Qubits =",num_qubits)
num_angles = num_layer * (4 * num_qubits - 1)
num_class = 10
learning_rate = LEARNING_RATE
sequence_length = 28
EPOCHS = EPOCHS
if num_qubits > 7 or num_qubits >= 3:
    EPOCHS = 5


def add_matchgate(qdev:tq.QuantumDevice, angles):
    ang = 0
    for i in range(num_qubits-1):
        qdev.rxx(params=angles[:, ang], wires=[i, i+1])
        ang += 1
        qdev.u3(params=angles[:, ang:ang+3], wires=i)
        ang += 3
    qdev.u3(params=angles[:, ang:ang+3], wires=num_qubits-1)
    

class RNN(torch.nn.Module):
    def __init__(self, input_size, num_angles):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.L = torch.nn.Linear(self.input_size, num_angles)

    def forward(self, input, qdev:tq.QuantumDevice):
        angles = self.L(input)
        for lay in range(num_layer):
            add_matchgate(qdev,angles[:,lay * (4 * num_qubits - 1) : (lay+1) * (4 * num_qubits - 1)])
        return qdev

class Classifier(torch.nn.Module):
    def __init__(self, num_qubits, num_class):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_class = num_class
        self.fc = nn.Linear(num_qubits, num_class)

    def forward(self, qdev: tq.QuantumDevice):
        class_list = []
        for i in range(self.num_qubits):
            meas = "I"*i + "Z" + "I"*(self.num_qubits - i - 1)
            class_list.append(tq.measurement.expval_joint_analytical(qdev, meas))
        res = nn.Softmax(dim=-1)
        stack = torch.stack(class_list, dim=-1)
        soft = res(stack)
        classLayer = self.fc(soft)
        return classLayer

model = RNN(sequence_length * sequence_length, num_angles)
classifier = Classifier(num_qubits, num_class)
optim = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model, classifier, inputs):
    hidden = tq.QuantumDevice(num_qubits, bsz=batch_size)
    inputs = inputs[:,0].view(inputs.size(0), -1)
    hidden = model.forward(inputs, hidden)
    output = classifier.forward(hidden)
    return output

def calculate_accuracy(model, classifier, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_tensor, true_output in tqdm.tqdm(dataset):
            output = evaluate(model, classifier, sentence_tensor)
            _, predicted = torch.max(output, 1)
            total += true_output.size(0)
            correct += (predicted == true_output).sum().item()
    return correct / total

import time 
import math
def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

start = time.time()
for epoch in range(EPOCHS):
    correct = 0
    total = 0
    start_epoch = time.time()
    for i, (sentence_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentence_tensor)
        loss = criterion(output, true_output)
        loss.backward()
        optim.step()

        _, predicted = torch.max(output, 1)
        total += true_output.size(0)
        correct += (predicted == true_output).sum().item()

    # Calculate accuracy on test dataset
    train_accuracy = correct / total
    print(f'Epoch: {epoch} Train Accuracy: {train_accuracy * 100:.2f}% - time start: {timeSince(start)}')

    # Calculate accuracy on test dataset
    test_accuracy = calculate_accuracy(model, classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')