from __future__ import print_function
import argparse
import os
import sys
import yaml
import random
import copy
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from dataset.ModelNetDataLoader10 import ModelNetDataLoader10
from dataset.ModelNetDataLoader import ModelNetDataLoader
from dataset.ShapeNetDataLoader import PartNormalDataset
from attack_utils import create_points_RS
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model/classifier'))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
    target = target[:,0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

parser = argparse.ArgumentParser()
# Data config
parser.add_argument(
    '--dataset', type=str, default='ModelNet40', help="dataset path")
parser.add_argument(
    '--split', type=int, default=1000, help='split the original dataset to get a small dataset possessed by the attacker')
parser.add_argument(
    '--feature_transform', action='store_true', help="use feature transform")
# Attack config
parser.add_argument(
    '--attack_dir', type=str, default='attack', help='attack folder')
parser.add_argument(
    '--SC', type=int, default=8, help='index of source class')
parser.add_argument(
    '--TC', type=int, default=35, help='index of target class')
# Optimization config
parser.add_argument(
    '--verbose', type=bool, default=False, help='print the details')
parser.add_argument(
    '--target_model', type=str, default='pointnet_cls', help='')
opt = parser.parse_args()

f = open('config.yaml')
config = yaml.safe_load(f)
opt.batch_size = config['batch_size']
opt.device = config['device']
opt.workers = config['workers']
opt.input_point_nums = config['input_point_nums']
opt.BD_NUM = config['PCBA']['BD_NUM']
opt.N = config['PCBA']['N']
opt.BD_POINTS = config['PCBA']['BD_POINTS']
opt.n_init = config['PCBA']['n_init']
opt.NSTEP = config['PCBA']['NSTEP']
opt.PI = config['PCBA']['PI']
opt.STEP_SIZE = config['PCBA']['STEP_SIZE']
opt.MOMENTUM = config['PCBA']['MOMENTUM']
opt.COST_INIT = config['PCBA']['COST_INIT']
opt.COST_MAX = config['PCBA']['COST_MAX']
opt.PATIENCE_UP = config['PCBA']['PATIENCE_UP']
opt.PATIENCE_DOWN = config['PCBA']['PATIENCE_DOWN']
opt.PATIENCE_CONVERGENCE = config['PCBA']['PATIENCE_CONVERGENCE']
opt.COST_UP_MULTIPLIER = config['PCBA']['COST_UP_MULTIPLIER']
opt.COST_DOWN_MULTIPLIER = config['PCBA']['COST_DOWN_MULTIPLIER']


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
# opt.manualSeed = random.randint(1, 10000)  # fix seed
opt.manualSeed = 2023 # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
opt.attack_dir = os.path.join(opt.attack_dir, opt.dataset)
opt.model_path = f'./model_surrogate/{opt.dataset}/{opt.target_model}.pth'
print(opt)

if not os.path.exists(opt.attack_dir):
    os.makedirs(opt.attack_dir)

# Load the surrogate classifier
classifier = load_model(opt)
classifier.load_state_dict(torch.load(opt.model_path))
classifier = classifier.to(opt.device)
classifier = classifier.eval()  


train_set = load_data(opt, 'train')

train_data = None
train_labels = None
attack_data = None
attack_labels = None
choices = np.random.choice(len(train_set), opt.split)

for c in tqdm(range(len(train_set))):
    if c in choices:
        if train_set[c][1] == opt.SC:
            if attack_data is None:
                attack_data = train_set[c][0][np.newaxis,:,:]
            else:
                attack_data = np.concatenate([attack_data, train_set[c][0][np.newaxis,:,:]], axis=0)
            if attack_labels is None:
                attack_labels = train_set[c][1]
            else:    
                attack_labels = np.concatenate([attack_labels, train_set[c][1]], axis=0)
    else:
        if train_data is None:
            train_data = train_set[c][0][np.newaxis,:,:]
        else:
            train_data = np.concatenate([train_data, train_set[c][0][np.newaxis,:,:]], axis=0)
        if train_labels is None:
            train_labels = train_set[c][1]
        else:    
            train_labels = np.concatenate([train_labels, train_set[c][1]], axis=0)        
    


attack_set = torch.utils.data.TensorDataset(torch.tensor(attack_data), torch.tensor(attack_labels).unsqueeze(-1))

# Get the subset of samples from the source class
pointoptloader = torch.utils.data.DataLoader(
    attack_set,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=4)

# Spatial location optimization
print('Spatial location optimization in progress...')
centers_best_global = None
dist_best_global = 1e10
for t in range(opt.n_init):
    centers = torch.zeros((opt.N, 3))
    while True:
        noise = torch.randn(centers.size()) * .5
        if torch.norm(noise).item() > 1.:
            break
    centers += noise

    grad_old = 0.
    ever_reached = False
    cost = opt.COST_INIT
    cost_up_counter = 0
    cost_down_counter = 0
    stopping_count = 0
    dist_best = 1e10
    centers_best = None
    for iter in tqdm(range(opt.NSTEP)):
        classifier.zero_grad()
        centers_trial = centers.clone()
        centers_trial = torch.unsqueeze(centers_trial, 0)
        centers_trial = centers_trial.to(opt.device)
        centers_trial.requires_grad = True
        (points, labels) = list(enumerate(pointoptloader))[0][1]
        labels = torch.ones_like(labels) * opt.TC
        points, labels = points.to(opt.device), labels.to(opt.device)
        centers_copies = centers_trial.repeat(len(points), 1, 1)
        points = torch.cat([points, centers_copies], dim=1)
        points = points.transpose(2, 1)
        pred = classifier(points)
        # Check if stopping criteria is satisfied
        posterior = torch.squeeze(torch.exp(pred), dim=0).detach().cpu()
        if opt.verbose and not ever_reached:
            print('iteration {}: mean target posterior: {}'.format(iter, torch.mean(posterior[:, opt.TC])))
        if torch.mean(posterior[:, opt.TC]) > opt.PI:
            ever_reached = True
        # Get gradient and update backdoor points
        loss = F.nll_loss(pred, labels.view(-1).long())
        # Involve the constaint term
        if ever_reached:
            dist = 0
            for n in range(opt.N):
                center_temp = centers_trial[0, n, :].repeat(points.size(-1)-opt.N, 1)
                for i in range(len(points)):
                    diff = points[i, :, :points.size(-1)-opt.N] - center_temp.transpose(1, 0)
                    diff_sqr = torch.square(diff)
                    dist_min = torch.min(torch.sum(diff_sqr, dim=0))
                    dist += dist_min
            dist = dist / (opt.N * len(points))
            loss += cost * dist

        loss.backward(retain_graph=True)
        grad = (1 - opt.MOMENTUM) * (centers_trial.grad / torch.norm(centers_trial.grad)) + opt.MOMENTUM * grad_old
        grad_old = grad
        centers -= opt.STEP_SIZE * torch.squeeze(grad.cpu(), dim=0)

        # Force stop
        if not ever_reached and iter >= int(opt.NSTEP * 0.1):
            break

        # Adjust the cost
        if ever_reached:
            if torch.mean(posterior[:, opt.TC]) >= opt.PI:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1
            # If the target class conf is smaller than PI for more than PATIENCE iterations, reduce the cost;
            # else, increase the cost
            if cost_up_counter >= opt.PATIENCE_UP and cost <= opt.COST_MAX:
                cost_up_counter = 0
                cost *= opt.COST_UP_MULTIPLIER
            elif cost_down_counter >= opt.PATIENCE_DOWN:
                cost_down_counter = 0
                cost /= opt.COST_DOWN_MULTIPLIER

            if opt.verbose:
                print('iteration {}: mean target posterior: {}; distance: {}; cost: {}; stopping: {}'.format(
                    iter, torch.mean(posterior[:, opt.TC]), dist, cost, stopping_count))

            # Stopping criteria
            if torch.mean(posterior[:, opt.TC]) >= opt.PI and dist < dist_best:
                dist_best = dist
                centers_best = copy.deepcopy(centers)
                stopping_count = 0
            else:
                stopping_count += 1

            if stopping_count >= opt.PATIENCE_CONVERGENCE:
                break
    if centers_best is not None:
        centers_best = centers_best.numpy()
    if dist_best < dist_best_global:
        centers_best_global = centers_best
        dist_best_global = dist_best

if centers_best_global is None:
    sys.exit('Optimization fails -- try more random initializations or reduce target confidence level.')

np.save(os.path.join(opt.attack_dir, f'centers.npy'), centers_best_global)


test_set = load_data(opt, 'test')


def create_attack_samples(idx, center, attack_dir, npoints, target, split, dataset):
    attack_data = []
    attack_labels = []
    points_inserted = []
    for i in range(len(idx)):
        points = dataset.__getitem__(idx[i])[0]
        if torch.is_tensor(points):
            points = points.detach().numpy()
        points_adv = create_points_RS(center=center, points=points, npoints=npoints)
        # Randomly delete points such that the resulting point cloud has the same size as a clean one
        ind_delete = np.random.choice(range(len(points)), len(points_adv), False)
        points = np.delete(points, ind_delete, axis=0)
        # Embed backdoor points
        points = np.concatenate([points, points_adv], axis=0)
        points_inserted.append(points_adv)
        attack_data.append(points)
        attack_labels.append(target)
    attack_data = np.asarray(attack_data)
    attack_labels = np.asarray(attack_labels)
    points_inserted = np.asarray(points_inserted)

    if split == 'train':
        # Save the indices of the clean images used for creating backdoor training images
        attack_data = np.concatenate([train_data, attack_data], axis=0)
        attack_labels = np.concatenate([train_labels, attack_labels], axis=0)        
        np.save(os.path.join(attack_dir, 'ind_train.npy'), choices)
    np.save(os.path.join(attack_dir, 'attack_data_{}.npy'.format(split)), attack_data)
    np.save(os.path.join(attack_dir, 'attack_labels_{}.npy'.format(split)), attack_labels)
    np.save(os.path.join(attack_dir, 'backdoor_pattern_{}.npy'.format(split)), points_inserted, allow_pickle=True)    


# Create backdoor samples
print('Creating backdoor samples...')
ind_train = [i for i in range(len(attack_labels))]
if len(ind_train) > opt.BD_NUM:
    ind_train = np.random.choice(ind_train, opt.BD_NUM, False)
create_attack_samples(ind_train, centers_best_global[0, :], opt.attack_dir, opt.BD_POINTS, opt.TC, 'train', attack_set)

ind_test = [i for i, data in enumerate(test_set) if data[1] == opt.SC]
create_attack_samples(ind_test, centers_best_global[0, :], opt.attack_dir, opt.BD_POINTS, opt.TC, 'test', test_set)
print('Success!!!')