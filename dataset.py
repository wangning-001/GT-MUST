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

class Dataset(data.Dataset):
    def __init__(self, data_path, data_mode):
        super(Dataset, self).__init__()
        self.image_height = 256
        self.image_width = 192
        self.radius = 5
        self.dataroot = data_path
        self.datamode = data_mode
        self.path = osp.join(self.dataroot, self.datamode)

        self.img_path = osp.join(self.path, 'image')
        self.img_data = self.load_list(self.img_path)

        self.parse_path = osp.join(self.path, 'image-parse')
        self.parse_data = self.load_list(self.parse_path)

        self.cloth_path = osp.join(self.path, 'cloth')
        self.cloth_data = self.load_list(self.cloth_path)

        self.cloth_mask_path = osp.join(self.path, 'cloth-mask')
        self.cloth_mask_data = self.load_list(self.cloth_mask_path)

        self.pose_path = osp.join(self.path, 'pose')
        self.names = self.load_name(self.img_data)

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.img_data[index])
            item = self.load_item(0)

        return item
#######
    # name_list = ['0: BG', '1: Hat', '2: Hair', '3: Glove', '4: Sunglasses',
    #              '5: UpperClothes', '6: Dress', '7: Coat', '8: Socks', '9: Pants',
    #              '10: Torso-skin', '11: Scarf', '12: Skirt', '13: Face', '14: Left-arm',
    #              '15: Right-arm', '16: Left-leg', '17: Right-leg', '18: Left-shoe', '19: Right-shoe']
#######
    def load_item(self, index):
        img = imread(self.img_data[index])
        parse = Image.open(self.parse_data[index])
        parse =np.array(parse)
        cloth = imread(self.cloth_data[index])
        cloth_mask = imread(self.cloth_mask_data[index])
        pose = self.load_pose(self.pose_path, self.names[index])
        name = self.names[index]

        return self.to_tensor(img), self.to_tensor(parse), self.to_tensor(cloth), self.to_tensor(cloth_mask), pose, name

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_name(self, datalist):
        names = []
        for i in range(len(datalist)):
            #names.append(datalist[i].split('\\')[-1][:-4]) # win
            names.append(datalist[i].split('/')[-1][:-4]) # linux
        return names

    def load_pose(self, path, name):
        pose_path = osp.join(path, name + '_keypoints.json')
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


    def load_list(self, path):
        if isinstance(path, str):
            if path[-3:] == "txt":
                line = open(path,"r")
                lines = line.readlines()
                file_names = []
                for line in lines:
                    file_names.append(self.path +line.split(" ")[0])
                return file_names
            if os.path.isdir(path):
                path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png'))
                path.sort()
                return path
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []
