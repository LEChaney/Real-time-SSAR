import numpy as np
import cv2
import torchvision.models.resnet as resnet
import torch
import os

from PIL import Image
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randrange
from spatial_transforms import Compose, Scale, CenterCrop, ToTensor, Normalize
from torchvision import transforms
from models.resnetl import resnetl10
from opts import parse_opts_offline
from model import generate_model
from mean import get_mean, get_std
from torch.nn import functional as F
from ssar.model.model import SSAR

NUM_CLASSES = 83
lstm_hidden = [None, None, None, None]

def pre_process_frame(frame, opt):
    # Convert from BGR opencv channel layout to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to pillow format for easy pre-processing
    frame = Image.fromarray(frame) 

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    spatial_transforms_det = Compose([
        Scale(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value),
        norm_method])

    # Use torchvision transforms for compatibility with SSAR model
    spatial_transforms_clf = transforms.Compose([
        transforms.Resize(opt.sample_size_clf),
        transforms.ToTensor(),
        transforms.Normalize(opt.mean_clf, opt.std_clf)])

    det_frame = spatial_transforms_det(frame)
    clf_frame = spatial_transforms_clf(frame)
    return det_frame, clf_frame


def test_on_video(video_path, detector, classifier, opt):
    cap = cv2.VideoCapture(video_path)

    # Figures setup
    figure = plt.figure()
    det_x_data, det_y_data = [], []
    clf_x_datas, clf_y_datas = [], []

    det_line, = plt.plot(det_x_data, det_y_data, '-')

    for i in range(NUM_CLASSES):
        clf_x_datas.append([])
        clf_y_datas.append([])
    clf_lines = []
    for i in range(NUM_CLASSES):
        clf_x_data = clf_x_datas[i]
        clf_y_data = clf_y_datas[i]
        clf_line, = plt.plot(clf_x_data, clf_y_data, '-')
        clf_lines.append(clf_line)

    det_line.axes.get_xaxis().set_visible(False)
    figure.gca().set_ylim([0.0, 1.1])

    def input_gen(opt):
        cv_ret, cv_frame = cap.read()
        if cv_ret and cv_frame is not None:
            # Initialize with duplicate of first frame
            frame_num = 0
            det_frame, clf_frame = pre_process_frame(cv_frame, opt)
            inputs_store_det = torch.stack([det_frame] * opt.sample_duration) 

        while cv_ret and cv_frame is not None:
            det_frame, clf_frame = pre_process_frame(cv_frame, opt)

            # Detector inputs
            inputs_store_det = torch.cat((inputs_store_det[1:], det_frame.unsqueeze(0))) # [C, H, W] -> [D, C, H, W] 
            inputs_det = inputs_store_det.permute(1, 0, 2, 3) # [D, C, H, W] -> [C, D, H, W]
            inputs_det = inputs_det.unsqueeze(0) # [C, D, H, W] -> [N, C, D, H, W]

            # Classifier inputs
            # Only need one frame at a time for classifier
            inputs_clf = clf_frame.unsqueeze(0) # [C, H, W] - > [N, C, H, W]
            inputs_clf = inputs_clf.cuda()

            yield frame_num, inputs_det, inputs_clf, cv_frame
            
            frame_num += 1
            cv_ret, cv_frame = cap.read()

    def plot_update(input_info):
        global lstm_hidden

        with torch.no_grad():
            frame_num, inputs_det, inputs_clf, cv_frame = input_info
            # cv_frame = inputs_det[0, :, -1].numpy() # [N, C, D, H, W] -> [C, H, W]
            # # std = np.array(opt.std).reshape(-1, 1, 1)
            # mean = np.array(opt.mean, dtype=np.float32).reshape(-1, 1, 1)
            # cv_frame = (cv_frame + mean) / 255 # Un-normalize
            # cv_frame = cv_frame.transpose(1, 2, 0) # [C, H, W] -> [H, W, C]
            # cv_frame = cv2.cvtColor(cv_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('frame', cv_frame)
            # cv2.waitKey(1)
            
            # print(inputs_det[0, :, -1, 0:4, 0:4])
            y_hat_det = F.softmax(detector(inputs_det), dim=1).cpu().numpy()[0, 1]

            # Toggle classifier using detector
            if y_hat_det > 0.5:
                mask, y_hat_clf, lstm_hidden = classifier(inputs_clf, lstm_hidden, get_mask=True)
                mask = (mask[0][0].cpu().numpy() + 127) / 255
                cv2.imshow('mask', mask)
                y_hat_clf = y_hat_clf.squeeze(dim=0)
                # print(y_hat_clf)
                cur_label_pred = torch.argmax(y_hat_clf, dim=0).item()
                y_hat_clf = F.softmax(y_hat_clf, dim=0).cpu().numpy()
                print(cur_label_pred + 1)
            else:
                lstm_hidden = [None, None, None, None]
                y_hat_clf = np.zeros(83)

            # Plot detector
            det_x_data.append(frame_num)
            det_y_data.append(y_hat_det)
            det_line.set_data(det_x_data, det_y_data)

            for i in range(NUM_CLASSES):
                clf_x_datas[i].append(frame_num)
                clf_y_datas[i].append(y_hat_clf[i])
                clf_lines[i].set_data(clf_x_datas[i], clf_y_datas[i])

            # Continous autoscale
            # figure.gca().relim()
            # figure.gca().autoscale_view()
            figure.gca().set_xlim([frame_num - 100, frame_num])
            figure.gca().set_ylim([0.0, 1.1])
            return [det_line] + clf_lines

    animation = FuncAnimation(figure, plot_update, input_gen(opt), interval=0, blit=True)

    plt.show()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Detector
    opt = parse_opts_offline()
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value)
    opt.std = get_std(opt.norm_value)

    detector, _ = generate_model(opt)

    if opt.resume_path:
        opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        detector.load_state_dict(checkpoint['state_dict'])

    detector.eval()

    # Classifier
    opt.sample_size_clf = (126, 224)
    opt.mean_clf = (0.485, 0.456, 0.406)
    opt.std_clf = (0.229, 0.224, 0.225)

    rnet = resnet.resnet18(False)
    classifier_weights = './ssar/weights/final_weights.pth'
    classifier = SSAR(ResNet=rnet, input_size=83, number_of_classes=83, batch_size=1, dropout=0).cuda()
    state = classifier.state_dict()
    state.update(torch.load(classifier_weights))
    classifier.load_state_dict(state)
    classifier.eval()

    test_on_video('../EgoGesture/Subject03/Scene4/Color/rgb2.avi', detector, classifier, opt)