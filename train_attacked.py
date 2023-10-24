from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import sys
import importlib
import numpy as np
import torch.nn as nn
from dataset.ModelNetDataLoader10 import ModelNetDataLoader10
from dataset.ModelNetDataLoader import ModelNetDataLoader
from dataset.ShapeNetDataLoader import PartNormalDataset
from model.classifier.pointnet_cls import feature_transform_regularizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))


def load_model(opt):
    MODEL = importlib.import_module(opt.target_model)
    classifier = MODEL.get_model(
        opt.num_class,
        normal_channel=opt.normal
    )
    classifier = classifier.to(opt.device)
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

#===================================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")
#! change
parser.add_argument(
    '--target_model', type=str, default='pointnet_cls', help='')
parser.add_argument('--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument('--output_dir', default='model_attacked',type=str)
opt = parser.parse_args()

f = open('config.yaml')
config = yaml.safe_load(f)
opt.batch_size = config['batch_size']
opt.device = config['device']
opt.workers = config['workers']
opt.input_point_nums = config['input_point_nums']
opt.nepoch = config['nepoch']

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
opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
opt.model_path = os.path.join(opt.output_dir, opt.dataset, 'checkpoints')
print(opt)

if not os.path.exists(opt.model_path):
    os.makedirs(opt.model_path)


testset = load_data(opt, 'test')
testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers)


# Load backdoor training samples
attack_data_train = np.load(os.path.join(opt.attack_dir, 'attack_data_train.npy'))
attack_labels_train = np.load(os.path.join(opt.attack_dir, 'attack_labels_train.npy'))
# Mix backdoor training samples with clean training samples

train_dataset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_train, dtype=torch.float32),torch.tensor(attack_labels_train).unsqueeze(-1))
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.workers)


# Load backdoor test samples 
attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))

attack_testset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_test, dtype=torch.float32),torch.tensor(attack_labels_test).unsqueeze(-1))
attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers))

print('train size: {}; test size: {}; attack test size: {}'.format(len(train_dataset),
                                                                   len(testset), len(attack_labels_test)))


classifier = load_model(opt)
optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
# optimizer = optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.6)
classifier.to(opt.device)

start_epoch = 0
for epoch in tqdm(range(start_epoch, opt.nepoch)):
    # print("epoch: {}".format(epoch))
    # Training
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
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
    scheduler.step()

    # Test accuracy on clean samples
    total_correct = 0
    total_testset = 0
    with torch.no_grad():
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
        print("test accuracy {} ({}/{})".format(total_correct / float(total_testset), total_correct, total_testset))

    # Test attack success rate
    total_correct = 0
    total_testset = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(attack_testloader)):
            points, targets = data_preprocess(data)
            points = points.transpose(2, 1)
            targets = targets.long()
            classifier = classifier.eval()
            pred = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(targets).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
        print("attack success rate {} ({}/{})".format(total_correct / float(total_testset), total_correct, total_testset))


# Final
total_correct = 0
total_testset = 0
with torch.no_grad():
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
    print("final test accuracy {} ({}/{})".format(total_correct / float(total_testset), total_correct, total_testset))

attack_total = 0
attack_correct = 0

with torch.no_grad():
    for i, data in tqdm(enumerate(attack_testloader)):
        points, targets = data_preprocess(data)
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        attack_correct += correct.item()
        attack_total += points.size()[0]
    print("final attack success rate {} ({}/{})".format(attack_correct / float(attack_total), attack_correct, attack_total))

with open(os.path.join(opt.output_dir,'data.txt'), 'a') as f:
    f.write(f'Training: {opt.target_model}, {opt.dataset}, ACR: {total_correct / float(total_testset)}, ASR: {attack_correct / float(attack_total)}\n')

torch.save(classifier.state_dict(),os.path.join(opt.model_path, opt.target_model+'.pth'))
print(f'save successfully! {opt.model_path}_{opt.target_model}')