'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as T
from torchvision.transforms import GaussianBlur
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet18
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer



imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]



imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Transform_single():
    def __init__(self, image_size, train, normalize=imagenet_norm):
        if train == True:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)

class SimCLRTransform():
    def __init__(self, image_size, mean_std=imagenet_mean_std, s=1.0):
        image_size = 224 if image_size is None else image_size
        self.transform = T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=0.5),
            # We blur the image 50% of the time using a Gaussian kernel. We randomly sample σ ∈ [0.1, 2.0], and the kernel size is set to be 10% of the image height/width.
            T.ToTensor(),
            T.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

def get_aug(image_size=32, train=True):

    if train == True:
        augmentation = SimCLRTransform(image_size)
    else:
        augmentation = Transform_single(image_size,train)

    return augmentation



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

term_width=400
#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim=256):
        super().__init__()
        hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x



class NTXent(nn.Module):
    """Wrap a module to get self.training member."""

    def __init__(self):
        super(NTXent, self).__init__()

    def forward(self, embedding1, embedding2, temperature=1):

        batch_size = embedding1.shape[0]
        LARGE_NUM = 1e9

        # normalize both embeddings

        embedding1 = F.normalize(embedding1,dim=-1)
        embedding2 = F.normalize(embedding2,dim=-1)

        embedding1_full = embedding1
        embedding2_full = embedding2
        masks = F.one_hot(torch.arange(batch_size), batch_size).cuda()
        labels = F.one_hot(torch.arange(batch_size), batch_size * 2).cuda()

        # Matmul-to-mask
        logits_aa = torch.matmul(embedding1, embedding1_full.T) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(embedding2, embedding2_full.T) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(embedding1, embedding2_full.T) / temperature
        logits_ba = torch.matmul(embedding2, embedding1_full.T) / temperature

        # Use our standard cross-entropy loss which uses log-softmax internally.
        # Concat on the feature dimension to provide all features for standard softmax-xent
        loss_a = F.cross_entropy(input=torch.cat([logits_ab, logits_aa], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss_b = F.cross_entropy(input=torch.cat([logits_ba, logits_bb], 1),
                                 target=torch.argmax(labels, -1),
                                 reduction="none")
        loss = loss_a + loss_b
        return torch.mean(loss)

def get_backbone(backbone=resnet18(pretrained=True), castrate=True):

    # backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone
class SimCLR(nn.Module):

    def __init__(self, backbone=resnet18(pretrained=True)):
        super().__init__()

        self.backbone = backbone
        self.projector = projection_MLP(backbone.output_dim)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        loss = NTXent()(z1, z2)
        return loss
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def linear_eval(path,num_epochs):

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=get_aug(train=False))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=get_aug(train=False))

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=0)

    model = get_backbone(resnet18(pretrained=True),castrate=True)
    classifier = nn.Linear(in_features=model.output_dim, out_features=100, bias=True).cuda()


    save_dict = torch.load(path, map_location='cpu')
    model = model.cuda()
    ls = {k[8:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')}
    model.load_state_dict(ls,strict=False)

    # define optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)


    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    global_progress = tqdm(range(0, num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader)

        for idx, (images, labels) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.cuda())

            preds = classifier(feature)

            loss = F.cross_entropy(preds, labels.cuda())

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.cuda())
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.cuda()).sum().item()
            acc_meter.update(correct / preds.shape[0])
    print(f'Accuracy = {acc_meter.avg * 100:.2f}')

