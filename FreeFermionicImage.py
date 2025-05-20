
import os
import sys

import torch
import torch.utils.data
from torch import nn
import tqdm

from torchvision import datasets
from torchvision.transforms import ToTensor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim import FFQuantumDevice

torch.manual_seed(0)
torch.set_printoptions(profile="full")
device = torch.device("cpu")

BATCH_SIZE = 128
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
train_dataset = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataset = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

num_layer = NUM_LAYER
num_qubits = NUM_QUBITS
print("Layer =",num_layer)
print("Qubits =",num_qubits)

num_angles = num_layer * (2 * num_qubits - 1)
num_class = 10
learning_rate = 0.001
if num_layer > 0:
    learning_rate = 0.005
if num_layer > 5:
    learning_rate = 0.0005
if num_layer > 20:
    learning_rate = 0.0001

learning_rate = 0.005
sequence_length = 28

class RNN(torch.nn.Module):
    def __init__(self, input_size, num_angles):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.num_angles = num_angles
        self.L = torch.nn.Linear(self.input_size, self.num_angles)

    def forward(self, input, circuit:FFQuantumDevice):
        angles = self.L(input)
        #print(angles.shape)

        ang = 0
        for _ in range(num_layer):
            
            circuit.rxx_layer(angles[:,ang:ang+num_qubits-1])
            ang += num_qubits - 1
            circuit.rz_layer(angles[:,ang:ang+num_qubits])
            ang += num_qubits

        return circuit

class Classifier(torch.nn.Module):
    def __init__(self, num_qubits, num_class):
        super(Classifier,self).__init__()
        self.num_qubits = num_qubits
        self.num_class = num_class
        self.fc = nn.Linear(num_qubits, num_class)

    def forward(self, circuit):
        meas = circuit.z_exp_all()        
        res = nn.Softmax(dim=-1)
        soft = res(meas)
        classLayer = self.fc(soft)
        return classLayer

model = RNN(sequence_length * sequence_length, num_angles).to(device)
classifier = Classifier(num_qubits, num_class).to(device)
optim = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def evaluate(model, classifier, inputs):
    hidden = FFQuantumDevice(num_qubits, batch_size, device=device)
    input = inputs[:,0].view(inputs.size(0), -1)
    circuit = model.forward(input, hidden)
    output = classifier.forward(circuit)
    return output

def calculate_accuracy(model, classifier, dataset):
    correct = 0
    total = 0
    with torch.no_grad():
        for sentence_tensor, true_output in tqdm.tqdm(dataset):
            output = evaluate(model, classifier, sentence_tensor.to(device))
            _, predicted = torch.max(output, 1)
            total += true_output.size(0)
            correct += (predicted.cpu() == true_output).sum().item()
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
    start_epoch = time.time()
    correct = 0
    total = 0
    for i, (sentence_tensor, true_output) in enumerate(tqdm.tqdm(train_dataset)):
        optim.zero_grad()
        output = evaluate(model, classifier, sentence_tensor.to(device))
        #print(true_output.shape)
        
        loss = criterion(output, true_output.to(device))
        #print(loss)
        loss.backward()
        optim.step()

        _, predicted = torch.max(output, 1)
        total += true_output.size(0)
        correct += (predicted.cpu() == true_output).sum().item()

    # Calculate accuracy on test dataset
    train_accuracy = correct / total
    print(f'Epoch: {epoch} Train Accuracy: {train_accuracy * 100:.2f}% - time start: {timeSince(start)}')

    # Calculate accuracy on test dataset
    test_accuracy = calculate_accuracy(model, classifier, test_dataset)
    print(f'Epoch: {epoch} Test Accuracy: {test_accuracy * 100:.2f}% - time start: {timeSince(start)}')