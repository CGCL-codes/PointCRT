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
     # [B, N, C]
    target = target[:,0] # [B]

    target = target.cuda()
    return points, target


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ModelNet40', help="dataset path")
    parser.add_argument('--N', type=int, default=100, help="dataset path")
    parser.add_argument('--target_model', type=str, default='pointnet_cls', help='')
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
    opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
    opt.model_path = os.path.join(opt.output_dir, opt.dataset, 'checkpoints', f'{opt.target_model}.pth')
    print(opt)

    classifier = load_model(opt)
    classifier.load_state_dict(torch.load(opt.model_path))
    classifier.to(opt.device)
    classifier = classifier.eval()
    
    testset = load_data(opt, 'test')
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=1,
            shuffle=False,
            num_workers=opt.workers)

    # Load backdoor test samples
    attack_data_test = np.load(os.path.join(opt.attack_dir, 'attack_data_test.npy'))
    attack_labels_test = np.load(os.path.join(opt.attack_dir, 'attack_labels_test.npy'))
    attack_testset =  torch.utils.data.TensorDataset(torch.tensor(attack_data_test, dtype=torch.float32),torch.tensor(attack_labels_test).unsqueeze(-1))
    attack_testloader = torch.utils.data.DataLoader(
            attack_testset,
            batch_size=1,
            shuffle=False,
            num_workers=int(opt.workers))


    test_dict = {}
    test_dict['spc'] = np.array([])
    test_dict['labels'] = np.array([])

    with torch.no_grad():
        for _, data in tqdm(enumerate(testloader)):
            points, targets = data_preprocess(data)
            targets = targets.long()
            spc = 0

            ori_pred = classifier(points.transpose(2,1)).data.max(1)[1]
            for n in [3,5,7,9,11]:            
                x = torch.clamp(points * n, -1, 1)
                pred = classifier(x.transpose(2,1)).data.max(1)[1]
               
                spc += int(pred.data==ori_pred.data)
            test_dict['spc'] = np.concatenate([test_dict['spc'], np.mean(np.array([spc]), keepdims=True)], axis=0) 
            test_dict['labels'] = np.concatenate([test_dict['labels'], targets.detach().cpu().numpy()], axis=0) 



    attack_dict = {}
    attack_dict['spc'] = []
    attack_dict['labels'] = np.array([])

    classifier = classifier.eval()
    with torch.no_grad():
        for _, data in tqdm(enumerate(attack_testloader)):
            points, targets = data_preprocess(data)
            targets = targets.long()
            spc = 0

            ori_pred = classifier(points.transpose(2,1)).data.max(1)[1]
            for n in [3,5,7,9,11]:            
                x = torch.clamp(points * n, -1, 1)
                pred = classifier(x.transpose(2,1)).data.max(1)[1]
               
                spc += int(pred.data==ori_pred.data)
            attack_dict['spc'] = np.concatenate([attack_dict['spc'], np.mean(np.array([spc]), keepdims=True)], axis=0) 
            attack_dict['labels'] = np.concatenate([attack_dict['labels'], targets.detach().cpu().numpy()], axis=0) 


    torch.save(test_dict,os.path.join(f'scaleup/{opt.output_dir}_{opt.dataset}_{opt.target_model}_test.pt'))
    torch.save(attack_dict, os.path.join(f'scaleup/{opt.output_dir}_{opt.dataset}_{opt.target_model}_attack.pt'))


        # spamwriter.writerow(attacks)


    """ with open(os.path.join(f'strip/{opt.output_dir}/{opt.dataset}',f'{opt.target_model}.txt'), 'a') as f:
        f.write(f"Corruption: {opt.corruption},\t Severity: {opt.severity}\n")
        f.write(f"Evaluation: {opt.target_model}, {opt.dataset}, ACC: {test_dict['ACC']}, ASR: {attack_dict['ASR']}\n") """

