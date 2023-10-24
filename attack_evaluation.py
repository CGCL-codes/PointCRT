from __future__ import print_function
import argparse
import os
import csv
import random
import numpy as np
import torch.utils.data
from tqdm import tqdm
import yaml
import sys
import importlib
import numpy as np
import torch.nn as nn
from utils.corruption import corrupt
from dataset.ModelNetDataLoader10 import ModelNetDataLoader10
from dataset.ModelNetDataLoader import ModelNetDataLoader
from dataset.ShapeNetDataLoader import PartNormalDataset
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# torch.backends.cudnn.enabled = False

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
    # print('Start Loading Dataset...')
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

    
    # print('Finish Loading Dataset...')
    return DATASET 

def data_preprocess(data):
    """Preprocess the given data and label.
    """
    points, target = data

    points = points.detach().numpy() # [B, N, C]
    target = target[:,0] # [B]

    target = target.cuda()
    return points, target

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--corruption', type=str, default='', help="dataset path")
parser.add_argument(
    '--severity', type=int, help='')
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
corrupt_class = corrupt(seed=opt.manualSeed)

opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
opt.model_path = os.path.join(opt.output_dir, opt.dataset, 'checkpoints',f'{opt.target_model}.pth')
print(opt)

classifier = load_model(opt)
classifier.load_state_dict(torch.load(opt.model_path))
classifier.to(opt.device)

corrupt = corrupt_class(opt.corruption)
testset = load_data(opt, 'test')
testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers)


num_pc = len(testset)
print('Num samples: {}'.format(num_pc))

# Load backdoor test samples
attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))
attack_testset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_test, dtype=torch.float32),torch.tensor(attack_labels_test).unsqueeze(-1))
attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers))

total_correct = 0
total_testset = 0

test_dict = {}
test_dict['pc'] = None
test_dict['labels'] = np.array([])
test_dict['pred'] = np.array([])

with torch.no_grad():
    for batch_id, data in tqdm(enumerate(testloader)):
        points, targets = data_preprocess(data)

        new_pc = []
        for i in range(points.shape[0]):
            corrupt_pc = corrupt(points[i], opt.severity)
            new_pc.append(corrupt_pc)
        new_pc = np.asarray(new_pc) 
        points = torch.tensor(new_pc, dtype=torch.float32).cuda()

        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

        if batch_id == 0:
            test_dict['pc'] = points.transpose(2,1).detach().cpu().numpy()
        else:            
            test_dict['pc'] = np.concatenate([test_dict['pc'], points.transpose(2,1).detach().cpu().numpy()], axis=0)
        test_dict['labels'] = np.concatenate([test_dict['labels'], targets.detach().cpu().numpy()], axis=0) 
        test_dict['pred'] = np.concatenate([test_dict['pred'], pred_choice.detach().cpu().numpy()], axis=0)
    print("accuracy {}".format(total_correct / float(total_testset)))
test_dict['ACC'] = total_correct / float(total_testset)



attack_correct = 0
attack_total = 0
attack_dict = {}
attack_dict['pc'] = None
attack_dict['labels'] = np.array([])
attack_dict['pred'] = np.array([])

with torch.no_grad():
    for batch_id, data in tqdm(enumerate(attack_testloader)):
        points, targets = data_preprocess(data)

        new_pc = []
        for i in range(points.shape[0]):
            corrupt_pc = corrupt(points[i], opt.severity)
            new_pc.append(corrupt_pc)
        new_pc = np.asarray(new_pc)
        points = torch.tensor(new_pc, dtype=torch.float32).cuda()

        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        attack_correct += correct.item()
        attack_total += points.size()[0]

        if batch_id == 0:
            attack_dict['pc'] = points.transpose(2,1).detach().cpu().numpy()
        else:
            attack_dict['pc'] = np.concatenate([attack_dict['pc'], points.transpose(2,1).detach().cpu().numpy()], axis=0)                    
        attack_dict['labels'] = np.concatenate([attack_dict['labels'], targets.detach().cpu().numpy()], axis=0) 
        attack_dict['pred'] = np.concatenate([attack_dict['pred'], pred_choice.detach().cpu().numpy()], axis=0)
    print("attack success rate {}".format(attack_correct / float(attack_total)))
attack_dict['ASR'] = attack_correct / float(attack_total)


if not os.path.exists(f'{opt.output_dir}/{opt.dataset}/{opt.target_model}'):
    os.makedirs(f'{opt.output_dir}/{opt.dataset}/{opt.target_model}')

# print(attack_dict['latent'].shape, attack_dict['pc'].shape)
# print(test_dict['latent'].shape, test_dict['pc'].shape)

torch.save(test_dict,os.path.join(f'{opt.output_dir}/{opt.dataset}/{opt.target_model}', f'{opt.target_model}_{opt.corruption}_{opt.severity}_test.pt'))
torch.save(attack_dict, os.path.join(f'{opt.output_dir}/{opt.dataset}/{opt.target_model}', f'{opt.target_model}_{opt.corruption}_{opt.severity}_attack.pt'))



""" with open(os.path.join(f'{opt.output_dir}/{opt.dataset}', f'{opt.target_model}.csv'), 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([opt.corruption, opt.severity, test_dict['ACC'], attack_dict['ASR']])
    # spamwriter.writerow(attacks)


with open(os.path.join(f'{opt.output_dir}/{opt.dataset}',f'{opt.target_model}.txt'), 'a') as f:
    f.write(f"Corruption: {opt.corruption},\t Severity: {opt.severity}\n")
    f.write(f"Evaluation: {opt.target_model}, {opt.dataset}, ACC: {test_dict['ACC']}, ASR: {attack_dict['ASR']}\n") """