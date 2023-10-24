# from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch.utils.data
from tqdm import tqdm
import yaml
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

    points = points # [B, N, C]
    target = target[:,0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

#===================================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--target_model', type=str, default='pointnet_cls', help='')
parser.add_argument('--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument('--output_dir', default='model_attacked',type=str)
opt = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
opt.manualSeed = 2023 # fix seed
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
opt.model_path = f'./{opt.output_dir}/{opt.dataset}/checkpoints/{opt.target_model}.pth'
print(opt)



testset = load_data(opt, 'test')
testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4)


num_classes = len(testset.classes)
# print('classes: {}'.format(num_classes))

# Load backdoor test images
# attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
# attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))

attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))
attack_testset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_test, dtype=torch.float32),torch.tensor(attack_labels_test).unsqueeze(-1))
attack_testloader = torch.utils.data.DataLoader(
        attack_testset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers))


# print('test size: {}; attack test size: {}'.format(len(testset), len(attack_labels_test)))

classifier = load_model(opt)
classifier.load_state_dict(torch.load(opt.model_path))
classifier.to(device)

total_correct = 0
total_testset = 0
with torch.no_grad():
    for i, data in enumerate(testloader):
        points, targets = data_preprocess(data)
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("accuracy {}".format(total_correct / float(total_testset)))


attack_correct = 0
attack_total = 0
with torch.no_grad():
    for i, data in enumerate(attack_testloader):
        points, targets = data_preprocess(data)
        points = points.transpose(2, 1)
        targets = targets.long()
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(targets).cpu().sum()
        attack_correct += correct.item()
        attack_total += points.size()[0]
    print("attack success rate {}".format(attack_correct / float(attack_total)))
    
with open(os.path.join('data.txt'), 'a') as f:
    f.write(f'Evaluation: {opt.target_model}, {opt.dataset}, ACC: {total_correct / float(total_testset)}, ASR: {attack_correct / float(attack_total)}\n')