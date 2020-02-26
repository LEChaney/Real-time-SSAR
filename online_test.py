import argparse
import time
import os
import glob 
import sys
import json
import shutil
import itertools
import numpy as np
import pandas as pd 
import csv
import torch
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from opts import parse_opts_online
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from torchvision import transforms
from target_transforms import ClassLabel
from dataset import get_online_data 
from utils import Logger, AverageMeter, LevenshteinDistancePlusAvgFramesEarly, Queue

import pdb
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

early_x_data = []
early_y_data = []
early_plot, = plt.plot([], [], 'bs-')
plt.xlabel('Early-detection threshold')
plt.ylabel('Average early detection time (frames)')


def weighting_func(x):
    return (1 / (1 + np.exp(-0.2*(x-9))))

def load_models(opt):
    opt.resume_path = opt.resume_path_det
    opt.pretrain_path = opt.pretrain_path_det
    opt.sample_duration = opt.sample_duration_det
    opt.model = opt.model_det
    opt.model_depth = opt.model_depth_det
    opt.modality = opt.modality_det
    opt.resnet_shortcut = opt.resnet_shortcut_det
    opt.n_classes = opt.n_classes_det
    opt.n_finetune_classes = opt.n_finetune_classes_det

    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)




    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_det.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    detector, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        detector.load_state_dict(checkpoint['state_dict'])

    print('Model 1 \n', detector)
    pytorch_total_params = sum(p.numel() for p in detector.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    # Reset parsed args
    opt = parse_opts_online()

    opt.resume_path = opt.resume_path_clf
    opt.pretrain_path = opt.pretrain_path_clf
    opt.sample_duration = opt.sample_duration_clf
    opt.model = opt.model_clf
    opt.model_depth = opt.model_depth_clf
    opt.modality = opt.modality_clf
    opt.resnet_shortcut = opt.resnet_shortcut_clf
    opt.n_classes = opt.n_classes_clf
    opt.n_finetune_classes = opt.n_finetune_classes_clf
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)

    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    if opt.model == 'ssar':
        opt.arch = opt.model
    else:
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts_clf.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)
    classifier, parameters = generate_model(opt)

    if opt.resume_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        classifier.load_state_dict(checkpoint['state_dict'])

    print('Model 2 \n', classifier)
    pytorch_total_params = sum(p.numel() for p in classifier.parameters() if
                               p.requires_grad)
    print("Total number of trainable parameters: ", pytorch_total_params)

    return detector, classifier

def main(clf_threshold_pre):
    print(f'Early-detection threshold: {clf_threshold_pre}')

    opt = parse_opts_online()

    detector,classifier = load_models(opt)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if opt.model_clf == 'ssar':
        opt.sample_size_clf = (126, 224)
        opt.mean_clf = (0.485, 0.456, 0.406)
        opt.std_clf = (0.229, 0.224, 0.225)

        spatial_transform_clf = transforms.Compose([
        transforms.Resize(opt.sample_size_clf),
        transforms.ToTensor(),
        transforms.Normalize(opt.mean_clf, opt.std_clf)])

    spatial_transform = Compose([
        Scale(112),
        CenterCrop(112),
        ToTensor(opt.norm_value), norm_method
        ])

    target_transform = ClassLabel()




    ## Get list of videos to test
    if opt.dataset == 'egogesture':
        subject_list = ['Subject{:02d}'.format(i) for i in [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]]
        test_paths = []
        for subject in subject_list:
            for x in glob.glob(os.path.join(opt.video_path,subject,'*/*/rgb*/')):
                test_paths.append(x)
    elif opt.dataset == 'nv':
        df = pd.read_csv(os.path.join(opt.video_path,'nvgesture_test_correct_cvpr2016_v2.lst'), delimiter = ' ', header = None)
        test_paths = []
        for x in df[0].values:
            test_paths.append(os.path.join(opt.video_path, x.replace('path:', ''), 'sk_color_all').replace(os.sep, '/'))

    # Figures setup
    # fig, ax = plt.subplots(nrows=6, ncols=1)

    # x_data, y_datas = [], []
    # lines = []
    # for j in range(6):
    #     if j != 0:
    #         ax[j].set_xlim(0, 400)
    #         ax[j].set_ylim(0, 1)
    #     y_datas.append([])
    #     lines.append([])
    #     for _ in range(opt.n_classes_clf):
    #         y_data = []
    #         y_datas[j].append(y_data)
    #         line, = ax[j].plot(x_data, y_data)
    #         lines[j].append(line)

    print('Start Evaluation')
    detector.eval()
    classifier.eval()

    levenshtein_accuracies = AverageMeter()
    frames_early_meter = AverageMeter()
    videoidx = 0
    for path in test_paths[4:]:
        path = os.path.normpath(path)
        if opt.dataset == 'egogesture':
            opt.whole_path = path.rsplit(os.sep, 4)[-4:]
            opt.whole_path = os.sep.join(opt.whole_path)
        elif opt.dataset == 'nv':
            opt.whole_path = path.split(os.sep, 3) # TODO: fix bad dependency on fixed depth file locations
            opt.whole_path = opt.whole_path[-1]
        
        videoidx += 1
        active_index = 0
        passive_count = 999
        active = False
        prev_active = False
        finished_prediction = None
        pre_predict = False

        cum_sum = np.zeros(opt.n_classes_clf,)
        cum_sum_unweighted = np.zeros(opt.n_classes_clf,)
        clf_selected_queue = np.zeros(opt.n_classes_clf,)
        det_selected_queue = np.zeros(opt.n_classes_det,)
        myqueue_det = Queue(opt.det_queue_size ,  n_classes = opt.n_classes_det)
        myqueue_clf = Queue(opt.clf_queue_size, n_classes = opt.n_classes_clf )


        print('[{}/{}]============'.format(videoidx,len(test_paths)))
        print(path)
        opt.sample_duration = max(opt.sample_duration_clf, opt.sample_duration_det)

        if opt.model_clf == 'ssar':
            test_data = get_online_data(
                opt, [spatial_transform, spatial_transform_clf], None, target_transform, modality='RGB')
        else:
            test_data = get_online_data(
                opt, spatial_transform, None, target_transform)

        test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=opt.batch_size,
                    shuffle=False,
                    num_workers=opt.n_threads,
                    pin_memory=True)


        results = []
        prev_best1 = opt.n_classes_clf

        if opt.model_clf == 'ssar':
            # Init recurrent state zero
            lstm_hidden = [None, None, None, None]

        for i, (inputs, targets) in enumerate(test_loader):
            if opt.model_clf == 'ssar':
                inputs, inputs_clf = inputs
            if not opt.no_cuda:
                targets = targets.cuda(non_blocking=True)
            ground_truth_array = np.zeros(opt.n_classes_clf +1,)
            with torch.no_grad():
                inputs = Variable(inputs)
                targets = Variable(targets)
                if opt.modality_det == 'RGB':
                    inputs_det = inputs[:,:3,-opt.sample_duration_det:,:,:]
                elif opt.modality_det == 'Depth':
                    inputs_det = inputs[:,-1,-opt.sample_duration_det:,:,:].unsqueeze(1)
                elif opt.modality_det =='RGB-D':
                    inputs_det = inputs[:,:,-opt.sample_duration_det:,:,:]
                
                # print(inputs_det[0, :, -1, 0:4, 0:4])
                outputs_det = detector(inputs_det)
                outputs_det = F.softmax(outputs_det,dim=1)
                outputs_det = outputs_det.cpu().numpy()[0].reshape(-1,)

                # enqueue the probabilities to the detector queue
                myqueue_det.enqueue(outputs_det.tolist())

                if opt.det_strategy == 'raw':
                    det_selected_queue = outputs_det
                elif opt.det_strategy == 'median':
                    det_selected_queue = myqueue_det.median
                elif opt.det_strategy == 'ma':
                    det_selected_queue = myqueue_det.ma
                elif opt.det_strategy == 'ewma':
                    det_selected_queue = myqueue_det.ewma
                

                prediction_det = np.argmax(det_selected_queue)
                prob_det = det_selected_queue[1]
                
                #### State of the detector is checked here as detector act as a switch for the classifier
                if  prediction_det == 1:
                    if opt.model_clf == 'ssar':
                        inputs_clf = Variable(inputs_clf)
                        if not opt.no_cuda:
                            inputs_clf = inputs_clf.cuda()
                        if opt.modality_clf == 'RGB':
                            inputs_clf = inputs_clf[:,:3,-1,:,:]
                        elif opt.modality_clf == 'Depth':
                            inputs_clf = inputs_clf[:,-1,-1,:,:].unsqueeze(1)
                        elif opt.modality_clf =='RGB-D':
                            inputs_clf = inputs_clf[:,:,-1,:,:]

                        outputs_clf, lstm_hidden = classifier(inputs_clf, lstm_hidden, get_lstm_state=True)
                    else:
                        if opt.modality_clf == 'RGB':
                            inputs_clf = inputs[:,:3,:,:,:]
                        elif opt.modality_clf == 'Depth':
                            inputs_clf = inputs[:,-1,:,:,:].unsqueeze(1)
                        elif opt.modality_clf =='RGB-D':
                            inputs_clf = inputs[:,:,:,:,:]

                        outputs_clf = classifier(inputs_clf)

                    outputs_clf = F.softmax(outputs_clf,dim=1)
                    outputs_clf = outputs_clf.cpu().numpy()[0].reshape(-1,)
                    
                    # Push the probabilities to queue
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count = 0

                    if opt.clf_strategy == 'raw':
                        clf_selected_queue = outputs_clf
                    elif opt.clf_strategy == 'median':
                        clf_selected_queue = myqueue_clf.median
                    elif opt.clf_strategy == 'ma':
                        clf_selected_queue = myqueue_clf.ma
                    elif opt.clf_strategy == 'ewma':
                        clf_selected_queue = myqueue_clf.ewma

                else:
                    if opt.model_clf == 'ssar':
                        # Reset recurrent state
                        lstm_hidden = [None, None, None, None]

                    outputs_clf = np.zeros(opt.n_classes_clf ,)
                    # Push the probabilities to queue
                    myqueue_clf.enqueue(outputs_clf.tolist())
                    passive_count += 1
            


            if passive_count >= opt.det_counter:
                active = False
            else:
                active = True

            # one of the following line need to be commented !!!!
            if active:
                active_index += 1
                cum_sum = ((cum_sum * (active_index-1)) + (weighting_func(active_index) * clf_selected_queue))/active_index # Weighted Aproach
                cum_sum_unweighted = ((cum_sum_unweighted * (active_index-1)) + (1.0 * clf_selected_queue))/active_index #Not Weighting Aproach 

                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if float(cum_sum[best1]- cum_sum[best2]) > clf_threshold_pre:
                    finished_prediction = True
                    pre_predict = True

            else:
                active_index = 0

            # Visualize
            # x_data.append(i)
            # y_datas[1][0].append(prob_det)
            # lines[1][0].set_xdata(x_data)
            # lines[1][0].set_ydata(y_datas[1][0])
            # for j in range(opt.n_classes_clf):
            #     y_datas[2][j].append(cum_sum[j])
            #     y_datas[3][j].append(cum_sum_unweighted[j])
            #     y_datas[4][j].append(clf_selected_queue[j] if active else 0)
            #     for k in range(2, 5):
            #         lines[k][j].set_xdata(x_data)
            #         lines[k][j].set_ydata(y_datas[k][j])
            # for k in range(1, 6):
            #     ax[k].set_xlim(i - 400, i)
            # mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, -1)
            # img = inputs_det[0, :, -1].permute(1, 2, 0).cpu().numpy() + mean
            # img = img.astype(int)
            # if i == 0:
            #     im_plt = ax[0].imshow(img)
            # else:
            #     im_plt.set_data(img)
            # plt.draw()
            # plt.pause(0.001)

            if active == False and  prev_active == True:
                finished_prediction = True
            elif active == True and  prev_active == False:
                finished_prediction = False



            if finished_prediction == True:
                detection_frame = (i * opt.stride_len) + opt.sample_duration_clf
                best2, best1 = tuple(cum_sum.argsort()[-2:][::1])
                if pre_predict == True:  
                    if best1 != prev_best1:
                        if cum_sum[best1]>opt.clf_threshold_final:  
                            results.append((detection_frame,best1))
                            print( 'Early Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], detection_frame))                      
                else:
                    # raw_best = clf_selected_queue.argsort()[-1]
                    # results.append((detection_frame,raw_best))
                    # print( 'Late Detected - class : {} with prob : {} at frame {}'.format(raw_best, clf_selected_queue[raw_best], detection_frame))
                    if cum_sum[best1]>opt.clf_threshold_final:
                        if best1 == prev_best1:
                            if cum_sum[best1]>5:
                                results.append((detection_frame,best1))
                                print( 'Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], detection_frame))
                        else:
                            results.append((detection_frame,best1))
                            
                            print( 'Late Detected - class : {} with prob : {} at frame {}'.format(best1, cum_sum[best1], detection_frame))

                    prev_best1 = best1
                    finished_prediction = False

                # prev_best1 = best1
                # finished_prediction = False

                cum_sum = np.zeros(opt.n_classes_clf,)
                cum_sum_unweighted = np.zeros(opt.n_classes_clf,)

            if active == False and  prev_active == True:
                pre_predict = False
        
            prev_active = active

        if opt.dataset == 'egogesture':
            opt.video_path = os.path.normpath(opt.video_path)
            opt.whole_path = os.path.normpath(opt.whole_path)
            target_csv_path = os.path.join(opt.video_path, 
                                    'labels-final-revised1',
                                    opt.whole_path.rsplit(os.sep, 2)[0],
                                    'Group'+opt.whole_path.rsplit('.', 1)[0][-1] + '.csv').replace('Subject', 'subject')
            true_classes = []
            end_frames = []
            with open(target_csv_path) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV:
                    true_classes.append(int(row[0])-1)
                    end_frames.append(int(row[2]))
        elif opt.dataset == 'nv':
            true_classes = []
            with open('./annotation_nvGesture/vallistall.txt') as csvfile:
                readCSV = csv.reader(csvfile, delimiter=' ')
                for row in readCSV:
                    if row[0] == opt.whole_path:
                        if row[1] != '26' :
                            true_classes.append(int(row[1])-1)
        
        predicted = np.array(results)[:,1]
        detection_frames = np.array(results)[:,0]
        
        true_classes = np.array(true_classes)
        levenshtein_distance, avg_frames_early = LevenshteinDistancePlusAvgFramesEarly(true_classes, predicted, end_frames, detection_frames)
        levenshtein_accuracy = 1-(levenshtein_distance/len(true_classes))
        if levenshtein_distance <0: # Distance cannot be less than 0
            levenshtein_accuracies.update(0, len(true_classes))
        else:
            levenshtein_accuracies.update(levenshtein_accuracy, len(true_classes))
        frames_early_meter.update(avg_frames_early)

        
        print('predicted classes: \t',predicted)
        print('True classes :\t\t',true_classes)
        print('Levenshtein Accuracy = {} ({})'.format(levenshtein_accuracies.val, levenshtein_accuracies.avg))
        print(f'Average frames early = {frames_early_meter.val} ({frames_early_meter.avg})')
        
    print('Average Levenshtein Accuracy= {}'.format(levenshtein_accuracies.avg))

    print('-----Evaluation is finished------')

    early_x_data.append(clf_threshold_pre)
    early_y_data.append(frames_early_meter.avg)
    early_plot.set_xdata(early_x_data)
    early_plot.set_ydata(early_y_data)
    plt.annotate(f'{levenshtein_accuracies.avg * 100:.2f}', (clf_threshold_pre, frames_early_meter.avg), textcoords='offset pixels', xytext=(10, 10))
    plt.gca().relim()
    plt.gca().autoscale_view()
    plt.pause(0.0001)
    plt.draw()

if __name__ == '__main__':
    for clf_threshold_pre in [0.6, 0.4, 0.2]:
        main(clf_threshold_pre)
    plt.show()