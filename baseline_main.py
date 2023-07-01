'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import *

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False,
                    help='resume from checkpoint')
args = parser.parse_args()

model_path = "checkpoint/2.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])





trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=get_aug(train=True))

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=get_aug(train=False))

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)



# Model
print('==> Building model..')

net = SimCLR(get_backbone()).to(device)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/1.pth')
    net.load_state_dict(checkpoint['state_dict'])


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)





# Training
writer = SummaryWriter('./path/to/log')
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    for inputs, targets in tqdm(trainloader):
        input_1,input_2 = inputs
        input_1, input_2, targets = input_1.to(device), input_2.to(device),targets.to(device)
        optimizer.zero_grad()
        loss = net(input_1,input_2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        total += targets.size(0)
        batch_idx += 1

    print("train_acc:", total / (trainloader.sampler.num_samples))
    writer.add_scalar('Loss_train',train_loss/(batch_idx+1), epoch)  # tensorboard train loss

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    writer.add_scalar('Loss_test', test_loss / (batch_idx + 1), epoch)  # tensorboard train loss
    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('Acc', acc, epoch)  # tensorboard acc
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def main():


    for epoch in tqdm(range(start_epoch, start_epoch+50)):

        train(epoch)


    torch.save({
        'epoch': epoch+1,
        'state_dict':net.state_dict()
    }, model_path)

    print(f"Model saved to {model_path}")

    linear_eval(model_path,num_epochs=20)

if __name__=='__main__':
    main()
