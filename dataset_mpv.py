import os
import os.path as osp
import glob
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from imageio import imread
import json
import linecache

class Dataset_MPV(data.Dataset):
    def __init__(self, data_path, data_mode):
        super(Dataset_MPV, self).__init__()
        self.image_height = 256
        self.image_width = 192
        self.radius = 5
        self.dataroot = data_path
        self.datamode = data_mode

        file_path = 'data_{:s}.txt'.format(self.datamode)
        file_path = osp.join(self.dataroot,file_path)
        self.dataset_size = len(open(file_path).readlines())

        self.img_id = []
        self.cloth_id = []
        for i in range(self.dataset_size):
            im_name, cm_name, c_name, parse_name, pose_name = linecache.getline(file_path, i + 1).strip().split()
            im_id = im_name.split('/')[1].split('.jpg')[0]
            c_id = c_name.split('/')[1].split('.jpg')[0]
            self.img_id.append(im_id)
            self.cloth_id.append(c_id)

        self.img_path = osp.join(self.dataroot, 'all')
        self.parse_path = osp.join(self.dataroot, 'all_parsing')
        self.pose_path = osp.join(self.dataroot, 'all_person_clothes_keypoints')

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.img_id[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        
        sub_dir_img = self.img_id[index].split('-')[0]
        sub_dir_c = self.cloth_id[index].split('-')[0]

        self.img_name = osp.join(self.img_path, sub_dir_img, self.img_id[index] + '.jpg')
        self.parse_name = osp.join(self.parse_path, sub_dir_img, self.img_id[index] + '.png')
        self.cloth_name = osp.join(self.img_path, sub_dir_c, self.cloth_id[index] + '.jpg')
        self.cloth_mask_name = osp.join(self.img_path, sub_dir_c, self.cloth_id[index] + '_mask.jpg')
        self.pose_name = osp.join(self.pose_path, sub_dir_img, self.img_id[index] + '_keypoints.json')

#        self.img_name = osp.join(self.img_path, self.img_id[index] + '.jpg')
#        self.parse_name = osp.join(self.parse_path, self.img_id[index] + '.png')
#        self.cloth_name = osp.join(self.img_path, self.cloth_id[index] + '.jpg')
#        self.cloth_mask_name = osp.join(self.img_path, self.cloth_id[index] + '_mask.jpg')
#        self.pose_name = osp.join(self.pose_path, self.img_id[index] + '_keypoints.json')

        img = imread(self.img_name)
        parse = Image.open(self.parse_name)
        parse = np.array(parse)
        cloth = imread(self.cloth_name)
        cloth_mask = imread(self.cloth_mask_name)
        pose = self.load_pose(self.pose_name)
        name = self.cloth_id[index]

        return self.to_tensor(img), self.to_tensor(parse), self.to_tensor(cloth), self.to_tensor(cloth_mask), pose, name

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_pose(self, pose_path):
        '''
        def add_name(name):
            return path + '\\' + name + '_keypoints.json' # win
            # return path + '/' + name + '_keypoints.json' # linux
        poses_path = list(map(lambda name: add_name(name), names))
        '''
        with open(pose_path, 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1,3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.image_height, self.image_width)
        im_pose = Image.new('L', (self.image_width, self.image_height))
        pose_draw = ImageDraw.Draw(im_pose)
        r = self.radius
        for i in range(point_num):
            one_map = Image.new('L', (self.image_width, self.image_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i,0]
            pointy = pose_data[i,1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
                pose_draw.rectangle((pointx-r, pointy-r, pointx+r, pointy+r), 'white', 'white')
            one_map = F.to_tensor(one_map.convert('RGB')).float()
            pose_map[i] = one_map[0]
        Pose_tensor = pose_map
        return Pose_tensor

