import os
import csv
import torch
import numpy as np
import argparse
from sklearn import metrics
import xgboost
import random
from sklearn.model_selection import train_test_split

def load_results(opt):
    
    for file in ['attack', 'test']: 
        all_dict =  {}
        for c in corruptions:
            all_dict[c] = {}
            all_dict[c]['pred'] = []
            all_dict[c]['labels'] = []
            if file == 'attack':
                all_dict[c]['ASR'] = []
            else:
                all_dict[c]['ACC'] = []
            for s in range(1, 6):
                    path = os.path.join(root, opt.target_model, f'{opt.target_model}_{c}_{s}_{file}.pt')
                    dict = torch.load(path)     
                    all_dict[c]['pred'].append(dict['pred'])
                    all_dict[c]['labels'].append(dict['labels']) 

                    if file == 'attack':
                        all_dict[c]['ASR'].append(dict['ASR'])
                    else:
                        all_dict[c]['ACC'].append(dict['ACC'])

        torch.save(all_dict, os.path.join(root, f'{opt.target_model}', f'results_{file}.pt'))    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, default='ModelNet40', choices=['ModelNet10', 'ModelNet40', 'ShapeNetPart'], help="dataset path")
    parser.add_argument('--ratio', type=float, default=0.9)
    parser.add_argument(
        '--target_model', type=str, default='pointnet_cls', help='')
    parser.add_argument('--result_dir', default='model_attacked')
    parser.add_argument('--r', action='store_true')
    opt = parser.parse_args()


    root = os.path.join(opt.result_dir, opt.dataset)


    corruptions = ["background", "cutout", "density", "density_inc", "distortion", \
                    "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", \
                    "original", "rotation","scale", "shear","uniform", "upsampling", 
                      "ufsampling" ]
    
    corruptions1 = ["cutout", "density", "density_inc", "ufsampling" ]
    
    corruptions2 = ["background", "gaussian", "impulse", "uniform", "upsampling"]
    
    corruptions3 = ["distortion", "distortion_rbf", "distortion_rbf_inv", "rotation","scale", "shear"]
    # random.shuffle(corruptions1)
    # random.shuffle(corruptions2)
    # random.shuffle(corruptions3)

    # corruptions = corruptions1[:3] + corruptions2[:3] + corruptions3[:]

    S_max = 5
    labels = []
    cor_s_list = []

    if opt.r:
        load_results(opt)

    corruptions.remove('original')     

    for file in ['attack', 'test']:  
        results = torch.load(os.path.join(root, f'{opt.target_model}', f'results_{file}.pt'))
        original_pred = results['original']['pred']
        temp = []
        for i in range(len(original_pred[0])):
            cor_s = []
            for c in corruptions:   
                flag = 0 
                for s in range(S_max):
                    pred = results[c]['pred'][s]
                    if int(pred[i]) != int(original_pred[0][i]):
                        cor_s.append(s+1)
                        flag = 1
                        break
                if flag == 0:
                    cor_s.append(S_max+1)
            cor_s = np.array(cor_s)
            cor_s_list.append(cor_s)


            if file == 'attack':
                labels.append(1)
            else:
                labels.append(0)

    crs_arr = np.asarray(cor_s_list)
    labels = np.asarray(labels)

    #! ===========================================================
    x_train, x_test, y_train, y_test = train_test_split(crs_arr, labels, test_size=opt.ratio, stratify=labels, random_state=2023)
    params = {'learning_rate': 0.05, 'n_estimators': 100, 'max_depth': 5, 'seed': 2023,
                        'subsample': 0.8, 'colsample_bytree': 0.7, 'n_jobs': 8, 
                        # 'gpu_id': 3,'tree_method': 'gpu_hist'
                        }                 
    classifier = xgboost.XGBClassifier(**params)
    classifier.fit(x_train, y_train)    
                    
    fpr, tpr, thresholds = metrics.roc_curve(y_test, classifier.predict_proba(x_test)[:,1])
    f1_scores = []
    for th in thresholds:
        f1_score = metrics.f1_score(labels, classifier.predict(crs_arr), average='micro')
        f1_scores.append(f1_score)
    f1_score = np.max(f1_scores)
    roc_auc = metrics.auc(fpr, tpr)
    print('%-20s%-15s%-20s%-20s%-10s'%(opt.result_dir,opt.dataset,opt.target_model,f1_score, roc_auc))
    #! ===========================================================  

    
    with open(os.path.join(f'results.csv'), 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([opt.dataset, opt.result_dir, opt.target_model, f1_score, roc_auc])        
    