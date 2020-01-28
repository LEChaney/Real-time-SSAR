from PIL import Image
from socket import gethostname
from torch.utils.data import Dataset
from data.data import check_and_split_data
from torchvision import transforms

import datetime
import glob
import numpy as np
import os
import pickle
import random
import skimage.util as ski_util
import torch


class EgoGestData(Dataset):
    def __init__(self, directory, hostname, image_transform=None, mask_transform=None):
        self.directory = directory
        self.hostname = hostname
        self.filelist = []
        self.labels = np.array(0)
        self.initialise_filelist()
        self.remove_zero_labels()
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def initialise_filelist(self):
        path = self.directory + 'labels-final-revised1'
        meta_data_folder = self.directory + '/.meta_data/'
        meta_data_file = meta_data_folder + "/{}_{}_meta_data.pkl".format("EgoGestData", self.hostname)
        print(meta_data_file)
        if os.path.exists(meta_data_folder):
            if os.path.exists(meta_data_file):
                with open(meta_data_file, 'rb') as f:
                    print('{}: Opening file list from saved metadata folder'.format(datetime.datetime.now().time()))
                    meta_data = pickle.load(f)
                    self.filelist = meta_data['filelist']
                    self.labels = meta_data['labels']
                    print('{}: Retrieved file list from saved metadata folder'.format(datetime.datetime.now().time()))
                    return
        else:
            os.mkdir(meta_data_folder)
        print('{}: Creating file list '.format(datetime.datetime.now().time()))
        subjects = sorted(os.listdir(path))
        for subject in subjects:
            if subject == '.DS_Store':
                continue
            subject_path = path + '/' + subject
            scenes = sorted(os.listdir(subject_path))
            for scene in scenes:
                if scene == '.DS_Store':
                    continue
                scene_path = subject_path + '/' + scene
                partitions = sorted(os.listdir(scene_path))
                for partition in partitions:
                    if partition == '.DS_Store':
                        continue
                    partition_id = int(partition[5])
                    partition_path = self.directory + '/' + subject + '/' + scene + '/Color/rgb' + str(
                        partition_id) + '/'
                    files = sorted(glob.glob(partition_path + '*.jpg'))
                    labels_file = scene_path + '/' + partition
                    self.update_labels(labels_file, len(files))
                    self.filelist.extend(files)
        self.labels = self.labels[1:]

        with open(meta_data_file, 'wb') as f:
            meta_data = {'filelist': self.filelist, 'labels': self.labels}
            pickle.dump(meta_data, f)
            print('{}: Saved {} file list to metadata folder'.format(datetime.datetime.now().time(), meta_data_file))

    def update_labels(self, labels_file, num_labels):
        labels = np.zeros(num_labels)
        for line in open(labels_file, 'r'):
            label, start_id, end_id = line.split(',')
            label, start_id, end_id = int(label), int(start_id), int(end_id)
            labels[start_id - 1:end_id - 1] = label - 1
        self.labels = np.append(self.labels, labels)

    def remove_zero_labels(self):
        new_labels = []
        new_files = []
        for i in range(len(self.labels)):
            if self.labels[i] != 0:
                new_labels.append(self.labels[i])
                new_files.append(self.filelist[i])
        self.labels = new_labels
        self.filelist = new_files

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        image_name = self.filelist[idx]
        mask_name = image_name.replace('Color/rgb', 'Depth/depth')
        image = Image.open(image_name).convert("RGB")
        mask = Image.open(mask_name).convert("L").resize((224, 126), Image.BILINEAR)
        threshold = 10
        mask = mask.point(lambda p: p > threshold and 255)

        modes = ['none', 'poisson', 'gaussian', 's&p']
        mode = modes[random.randint(0, 3)]

        if mode != 'none':
            im_as_array = ski_util.random_noise(np.asarray(image), mode, seed=None, clip=True)
            image = Image.fromarray(np.uint8(im_as_array * 255))

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = torch.squeeze(mask)

        sample = {'image': image, 'mask': mask, 'label': self.labels[idx], 'name': image_name}

        return sample


class EgoGestDataSequence(Dataset):
    def __init__(self, directory, hostname, image_transform=None, mask_transform=None):
        self.directory = directory
        self.hostname = hostname
        self.gesture_list = []
        self.initialise_gesture_list()
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def initialise_gesture_list(self):
        path = self.directory + 'labels-final-revised1'
        meta_data_folder = self.directory + '/.meta_data/'
        meta_data_file = meta_data_folder + "/{}_{}_sequence_meta_data.pkl".format("EgoGestData", self.hostname)
        print(meta_data_file)
        if os.path.exists(meta_data_folder):
            if os.path.exists(meta_data_file):
                with open(meta_data_file, 'rb') as f:
                    print('{}: Opening file list from saved metadata folder'.format(datetime.datetime.now().time()))
                    meta_data = pickle.load(f)
                    self.gesture_list = meta_data['gesture_list']
                    print('{}: Retrieved file list from saved metadata folder'.format(datetime.datetime.now().time()))
                    return
        else:
            os.mkdir(meta_data_folder)
        subjects = sorted(os.listdir(path))
        for subject in subjects:
            if subject == '.DS_Store':
                continue
            subject_path = path + '/' + subject
            scenes = sorted(os.listdir(subject_path))
            for scene in scenes:
                if scene == '.DS_Store':
                    continue
                scene_path = subject_path + '/' + scene
                partitions = sorted(os.listdir(scene_path))
                for partition in partitions:
                    if partition == '.DS_Store':
                        continue
                    partition_id = int(partition[5])
                    partition_path = self.directory + '/' + subject + '/' + scene + '/Color/rgb' + str(
                        partition_id) + '/'
                    files = sorted(glob.glob(partition_path + '*.jpg'))
                    labels_file = scene_path + '/' + partition
                    self.update_gesture_list(labels_file, files)

        with open(meta_data_file, 'wb') as f:
            meta_data = {'gesture_list': self.gesture_list}
            pickle.dump(meta_data, f)
            print('{}: Saved {} file list to metadata folder'.format(datetime.datetime.now().time(), meta_data_file))

    def update_gesture_list(self, labels_file, files):
        for line in open(labels_file, 'r'):
            label, start_id, end_id = line.split(',')
            label, start_id, end_id = int(label), int(start_id), int(end_id)
            gesture_files = files[start_id-1:end_id - 1]
            gesture = (label-1, gesture_files)
            self.gesture_list.append(gesture)

    def __len__(self):
        return len(self.gesture_list)

    def __getitem__(self, idx):

        gesture = self.gesture_list[idx]
        image_names = gesture[1]
        mask_names = []
        for image_name in image_names:
            mask_names.append(image_name.replace('Color/rgb', 'Depth/depth'))
        num_images = len(image_names)

        images = torch.ones((num_images, 3, 126, 224))
        masks = torch.ones(num_images, 126, 224)
        for image_name, mask_name, i in zip(image_names, mask_names, range(num_images)):
            image = Image.open(image_name).convert("RGB").resize((224, 126), Image.BILINEAR)
            mask = Image.open(mask_name).convert("L").resize((224, 126), Image.BILINEAR)
            threshold = 10
            mask = mask.point(lambda p: p > threshold and 255)
            if self.image_transform:
                image = self.image_transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)
                mask = torch.squeeze(mask)
            images[i, :, :, :] = image
            masks[i, :, :] = mask

        sample = {'images': images, 'masks': masks, 'label': gesture[0],
                  'img_name': image_names, 'msk_name': mask_names, 'length': num_images}

        return sample