from __future__ import print_function
import argparse
import os
import random
import yaml
import torch.optim as optim
import torch.utils.data
from model.classifier.pointnet_cls import feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm

import sys
import importlib
import numpy as np
import torch.nn as nn
from dataset.ModelNetDataLoader10 import ModelNetDataLoader10
from dataset.ModelNetDataLoader import ModelNetDataLoader
from dataset.ShapeNetDataLoader import PartNormalDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))
def load_model(args):
    MODEL = importlib.import_module(args.target_model)
    classifier = MODEL.get_model(
        args.num_class,
        normal_channel=args.normal
    )
    classifier = classifier.to(args.device)
    classifier = nn.DataParallel(classifier)
    return classifier


def load_data(opt, split='test'):
    """Load the dataset from the given path.
    """
    print('Start Loading Dataset...')
    if opt.dataset == 'ModelNet40':
        DATASET = ModelNetDataLoader(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False
        )
    elif opt.dataset == 'ShapeNetPart':
        DATASET = PartNormalDataset(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False
        )
    elif opt.dataset == 'ModelNet10':
        DATASET = ModelNetDataLoader10(
            root=opt.data_path,
            npoints=opt.input_point_nums,
            split=split,
            normal_channel=False            
        )
    else:
        raise NotImplementedError

    
    print('Finish Loading Dataset...')
    return DATASET 

def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

parser = argparse.ArgumentParser()
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=250, help='number of epochs to train')
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")

#! change
parser.add_argument(
    '--target_model', type=str, default='pointnet_cls', help='')

opt = parser.parse_args()

f = open('config.yaml')
config = yaml.safe_load(f)
opt.batch_size = config['batch_size']
opt.device = config['device']
opt.workers = config['workers']
opt.input_point_nums = config['input_point_nums']

if opt.dataset == 'ModelNet40':
    opt.num_class = 40
    opt.data_path = config['ModelNet_path']
elif opt.dataset == 'ShapeNetPart':
    opt.num_class = 16
    opt.data_path = config['ShapeNetPart_path']
elif opt.dataset == 'ModelNet10':
    opt.num_class = 10
    opt.data_path = config['ModelNet_path']
opt.normal =False


opt.manualSeed = 2023  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
opt.model_path = f'./model_surrogate/{opt.dataset}'
print(opt)

if not os.path.exists(opt.model_path):
    os.makedirs(opt.model_path)

trainset = load_data(opt, 'train')
testset = load_data(opt, 'test')
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=4)


# Get a subset of the experiment dataset
data = None
labels = None
choices = np.random.choice(len(trainset), opt.split)
for c in choices:
    if data is None:
        data = trainset[c][0][np.newaxis,:,:]
    else:
        data = np.concatenate([data, trainset[c][0][np.newaxis,:,:]], axis=0)
    if labels is None:
        labels = trainset[c][1]
    else:    
        labels = np.concatenate([labels, trainset[c][1]], axis=0)

train_dataset = torch.utils.data.TensorDataset(torch.tensor(data),torch.tensor(labels).unsqueeze(-1))
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=4)

print('classes: {}'.format(opt.num_class))
print('train size: {}; test size: {}'.format(len(trainloader), len(testloader)))



#? classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

classifier = load_model(opt)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
classifier.to(opt.device)


for epoch in tqdm(range(opt.nepoch)):
    print("epoch {}".format(epoch))
    for i, data in enumerate(trainloader):
        points, targets = data_preprocess(data)
        points = points.transpose(2, 1)
        targets = targets.long()
        optimizer.zero_grad()
        classifier = classifier.train()

        if opt.target_model == 'pointnet_cls':
            pred, trans, trans_feat = classifier(points, is_train=True)
            loss = F.nll_loss(pred, targets)
            loss += feature_transform_regularizer(trans_feat) * 0.001
        else:
            pred = classifier(points)
            loss = F.cross_entropy(pred, targets)
        loss.backward()
        optimizer.step()
        GRAD = 0
        for name, params in classifier.named_parameters():
            # print(name)
            # print(params)
            # print(params.grad)
            GRAD += torch.mean(abs(params.grad.data))
        print(GRAD)
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        if i == 20:
            sys.exit(-1)
    scheduler.step()

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testloader)):
        points, targets = data_preprocess(data)
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("test accuracy {}".format(total_correct / float(total_testset)))

total_correct = 0
total_testset = 0
for i, data in tqdm(enumerate(testloader)):
    points, targets = data_preprocess(data)
    points = points.transpose(2, 1)
    targets = targets.long()
    classifier = classifier.eval()
    pred = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(targets).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

print("final accuracy {}".format(total_correct / float(total_testset)))


with open(os.path.join(opt.model_path,'data.txt'), 'a') as f:
    f.write(f'{opt.target_model}, {opt.dataset}, {total_correct / float(total_testset)}\n')


torch.save(classifier.state_dict(),os.path.join(opt.model_path, opt.target_model+'.pth'))
