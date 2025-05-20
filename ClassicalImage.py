
import torch
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor

import torch.optim as optim

torch.manual_seed(0)

train_data = datasets.MNIST(
    root = './data',
    train = True,                         
    transform = ToTensor()            
)
test_data = datasets.MNIST(
    root = './data', 
    train = False, 
    transform = ToTensor()
)

batch_size = 128
trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

# parameters 
N_STEPS = 28
input_size = 28
hidden_size = 64
output_size = 10
epochs = 20

class classicalRNN(nn.Module):
    def __init__(self, batch_size, n_steps, input_size, hidden_size, output_size):
        super(classicalRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
    
        self.rnn = nn.RNN(self.input_size, self.hidden_size) 
        
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
    def init_hidden(self,):
        return (torch.zeros(1, self.batch_size, self.hidden_size))
        
    def forward(self, X):
        X = X.permute(1, 0, 2) 
        
        self.batch_size = X.size(1)
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.rnn(X, self.hidden)
        out = self.fc(self.hidden)
        
        return out.view(-1, self.output_size)


# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model instance
model = classicalRNN(batch_size, N_STEPS, input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def get_accuracy(logit, target, batch_size):
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()

import time 
import math
def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(epochs):  # loop over the dataset multiple times
    train_running_loss = 0.0
    train_acc = 0.0
    model.train()
    
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()
        model.hidden = model.init_hidden() 
        inputs, labels = data
        inputs = inputs.view(-1, 28,28) 
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_running_loss += loss.detach().item()
        train_acc += get_accuracy(outputs, labels, batch_size)
         
    model.eval()
    print_train_running_loss = train_running_loss/i
    print_train_acc = train_acc/i
    print(f'Epoch: {epoch} | Loss: {print_train_running_loss:.4f} | Train Accuracy: {print_train_acc:.2f}% | time since start: {timeSince(start)}')


    test_acc = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)

        outputs = model(inputs)

        test_acc += get_accuracy(outputs, labels, batch_size)
            
    print(f'Test Accuracy: {test_acc/i:.2f}')