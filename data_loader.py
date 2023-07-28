from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import random
import os
import numpy as np
from PIL import Image

class Base_Dataset(data.Dataset):
    def __init__(self, root, partition, target_ratio=0.0):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        self.target_ratio = target_ratio
        # self.target_ratio=0 no mixup
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])
        else:
            self.transformer = transforms.Compose([transforms.Resize(256),
                                                   transforms.CenterCrop(224),
                                                   transforms.ToTensor(),
                                                   normalize])

    def __len__(self):

        if self.partition == 'train':
            return int(min(sum(self.alpha), len(self.target_image)) / (self.num_class - 1))
        elif self.partition == 'test':
            return int(len(self.target_image) / (self.num_class - 1))

    def __getitem__(self, item):

        image_data = []
        label_data = []

        target_real_label = []
        class_index_target = []

        domain_label = []
        ST_split = [] # Mask of targets to be evaluated
        # select index for support class
        num_class_index_target = int(self.target_ratio * (self.num_class - 1))

        if self.target_ratio > 0:
            available_index = [key for key in self.target_image_list.keys() if len(self.target_image_list[key]) > 0
                               and key < self.num_class - 1]
            class_index_target = random.sample(available_index, min(num_class_index_target, len(available_index)))

        class_index_source = list(set(range(self.num_class - 1)) - set(class_index_target))
        random.shuffle(class_index_source)

        for classes in class_index_source:
            # select support samples from source domain or target domain
            # image = Image.open(random.choice(self.source_image[classes])).convert('RGB')
            source_images = np.array(self.source_image[classes])[:, 0]
            source_domains = np.array(self.source_image[classes])[:, 2]
            choiced_idx = random.choice(range(len(source_images)))
            image = Image.open(source_images[choiced_idx]).convert('RGB')
            d_label = int(source_domains[choiced_idx])

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            # domain_label.append(1)
            domain_label.append(d_label)
            ST_split.append(0)
            # target_real_label.append(classes)
        for classes in class_index_target:
            # select support samples from source domain or target domain
            # image = Image.open(random.choice(self.target_image_list[classes])).convert('RGB')
            target_images = np.array(self.target_image_list[classes])[:, 0]
            target_domains = np.array(self.target_image_list[classes])[:, 2]
            choiced_idx = random.choice(range(len(target_images)))
            image = Image.open(target_images[choiced_idx]).convert('RGB')
            d_label = int(target_domains[choiced_idx])

            if self.transformer is not None:
                image = self.transformer(image)
            image_data.append(image)
            label_data.append(classes)
            # domain_label.append(0)
            domain_label.append(d_label)
            ST_split.append(0)
            # target_real_label.append(classes)

        # adding target samples
        for i in range(self.num_class - 1):

            if self.partition == 'train':
                if self.target_ratio > 0:
                    index = random.choice(list(range(len(self.label_flag))))
                else:
                    index = random.choice(list(range(len(self.target_image))))
                # index = random.choice(list(range(len(self.label_flag))))
                target_image = Image.open(self.target_image[index]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(int(self.label_flag[index]))
                target_real_label.append(self.target_label[index])
                # domain_label.append(0)
                domain_label.append(int(self.target_domain[index]))
                ST_split.append(1)
            elif self.partition == 'test':
                # For last batch
                # if item * (self.num_class - 1) + i >= len(self.target_image):
                #     break
                target_image = Image.open(self.target_image[item * (self.num_class - 1) + i]).convert('RGB')
                if self.transformer is not None:
                    target_image = self.transformer(target_image)
                image_data.append(target_image)
                label_data.append(self.num_class)
                target_real_label.append(self.target_label[item * (self.num_class - 1) + i])
                # domain_label.append(0)
                domain_label.append(int(self.target_domain[item * (self.num_class - 1) + i]))
                ST_split.append(1)
        image_data = torch.stack(image_data)
        label_data = torch.Tensor(label_data).long()
        real_label_data = torch.tensor(target_real_label)
        domain_label = torch.tensor(domain_label)
        ST_split = torch.tensor(ST_split)
        return image_data, label_data, real_label_data, domain_label, ST_split

    def load_dataset(self):
        source_image_list = {key: [] for key in range(self.num_class - 1)}
        target_image_list = []
        target_label_list = []
        target_domain_list = []
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label, domain = line.split(' ')
                label = label.strip()
                if label == str(self.num_class-1):
                    continue
                source_image_list[int(label)].append((os.path.join(self.root, image_dir), int(label), int(domain)))

        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label, domain = line.split(' ')
                label = label.strip()
                target_image_list.append(os.path.join(self.root, image_dir))
                target_label_list.append(int(label))
                target_domain_list.append(int(domain))

        return source_image_list, target_image_list, target_label_list, target_domain_list


class Office_Dataset(Base_Dataset):

    def __init__(self, root, txt_root, partition, label_flag=None, source='A', target='W', target_ratio=0.0):
        super(Office_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(txt_root, src_name)
        self.target_path = os.path.join(txt_root, tar_name)
        self.num_known_classes = 20
        self.class_name = self.get_cls_names() + ["unk"]
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label, self.target_domain = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class
            self.label_flag = self.label_flag.long()

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append((self.target_image[i], self.label_flag[i].item(), self.target_domain[i]))

        if self.target_ratio > 0:
            self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in self.source_image.keys()]
        else:
            self.alpha_value = self.alpha

        self.alpha_value = np.array(self.alpha_value)
        self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def get_cls_names(self):
        results = {}
        with open('/nas/data/syamagami/GDA/data/Office31/amazon.txt', 'r') as f:
            lines = f.readlines()
        sps = [line.strip('\n').split(' ') for line in lines]
        for sp in sps:
            cls = sp[0].split('/')[2]
            results[cls] = int(sp[1])
        return list(results.keys())[:self.num_known_classes]

    def getFilePath(self, source, target):
        domain_info = {'A': 'amazon', 'W': 'webcam', 'D': 'dslr'}
        full_dset = f'{domain_info[source]}_{domain_info[target]}'
        src_name = f'{full_dset}/labeled.txt'
        tar_name = f'{full_dset}/unlabeled.txt'
        return src_name, tar_name

        # if source == 'A':
        #     src_name = 'amazon_src_list.txt'
        # elif source == 'W':
        #     src_name = 'webcam_src_list.txt'
        # elif source == 'D':
        #     src_name = 'dslr_src_list.txt'
        # else:
        #     print("Unknown Source Type, only supports A W D.")

        # if target == 'A':
        #     tar_name = 'amazon_tar_list.txt'
        # elif target == 'W':
        #     tar_name = 'webcam_tar_list.txt'
        # elif target == 'D':
        #     tar_name = 'dslr_tar_list.txt'
        # else:
        #     print("Unknown Target Type, only supports A W D.")

        # return src_name, tar_name

class Home_Dataset(Base_Dataset):
    def __init__(self, root, txt_root, partition, label_flag=None, source='A', target='R', target_ratio=0.0):
        super(Home_Dataset, self).__init__(root, partition, target_ratio)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join(txt_root, src_name)
        self.target_path = os.path.join(txt_root, tar_name)
        self.class_name = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                           'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                           'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                           'Fork', 'unk']
        self.num_class = len(self.class_name)

        self.source_image, self.target_image, self.target_label, self.target_domain = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append((self.target_image[i], self.label_flag[i].item(), self.target_domain[i]))

        # if self.target_ratio > 0:
        #     self.alpha_value = [len(self.source_image[key]) + len(self.target_image_list[key]) for key in
        #                         self.source_image.keys()]
        # else:
        #     self.alpha_value = self.alpha
        #
        # self.alpha_value = np.array(self.alpha_value)
        # self.alpha_value = (self.alpha_value.max() + 1 - self.alpha_value) / self.alpha_value.mean()
        # self.alpha_value = torch.tensor(self.alpha_value).float().cuda()

    def getFilePath(self, source, target):
        domain_info = {'A': 'Art', 'C': 'Clipart', 'P': 'Product', 'R': 'RealWorld'}
        full_dset = f'{domain_info[source]}_{domain_info[target]}'
        src_name = f'{full_dset}/labeled.txt'
        tar_name = f'{full_dset}/unlabeled.txt'
        return src_name, tar_name
        
        if source == 'A':
            src_name = 'art_source.txt'
        elif source == 'C':
            src_name = 'clip_source.txt'
        elif source == 'P':
            src_name = 'product_source.txt'
        elif source == 'R':
            src_name = 'real_source.txt'
        else:
            print("Unknown Source Type, only supports A C P R.")

        if target == 'A':
            tar_name = 'art_tar.txt'
        elif target == 'C':
            tar_name = 'clip_tar.txt'
        elif target == 'P':
            tar_name = 'product_tar.txt'
        elif target == 'R':
            tar_name = 'real_tar.txt'
        else:
            print("Unknown Target Type, only supports A C P R.")

        return src_name, tar_name


class Visda_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["bicycle", "bus", "car", "motorcycle", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])

class Visda18_Dataset(Base_Dataset):
    def __init__(self, root, partition, label_flag=None, target_ratio=0.0):
        super(Visda18_Dataset, self).__init__(root, partition, target_ratio)
        # set dataset info
        self.source_path = os.path.join(root, 'source_list_k.txt')
        self.target_path = os.path.join(root, 'target_list.txt')
        self.class_name = ["areoplane","bicycle", "bus", "car", "horse", "knife", "motorcycle", "person", "plant",
                           "skateboard", "train", "truck", 'unk']
        self.num_class = len(self.class_name)
        self.source_image, self.target_image, self.target_label = self.load_dataset()
        self.alpha = [len(self.source_image[key]) for key in self.source_image.keys()]
        self.label_flag = label_flag

        # create the unlabeled tag
        if self.label_flag is None:
            self.label_flag = torch.ones(len(self.target_image)) * self.num_class

        else:
            # if pseudo label comes
            self.target_image_list = {key: [] for key in range(self.num_class + 1)}
            for i in range(len(self.label_flag)):
                self.target_image_list[self.label_flag[i].item()].append(self.target_image[i])
