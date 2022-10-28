import os

import scipy.io
import scipy.misc
import numpy as np
from PIL import Image
import cv2

import mindspore.dataset as ds
from mindspore.communication.management import get_rank, get_group_size, init


import src.data.transforms as transforms
from .data_utils.moxing_adapter import sync_data


def read_mat_file(mode, root_dir, img_list):
 
    if mode == 'joints_train':
        # lspnet (14,3,10000)
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints_train.mat'))['joints_train']

    elif mode =='joints_test':
        # lsp (3,14,2000)
        mat_arr = scipy.io.loadmat(os.path.join(root_dir, 'joints_test.mat'))['joints_test']

    mat_arr[2] = np.logical_not(mat_arr[2])
    lms = mat_arr.transpose([2, 0, 1])
    kpts = mat_arr.transpose([2, 1, 0]).tolist()
    for kpt in kpts:
        for pt in kpt:
            pt[1],pt[0] = pt[0],pt[1]

    centers = []
    scales = []
    for idx in range(lms.shape[0]):
        im = Image.open(img_list[idx])
        w = im.size[0]
        h = im.size[1]

        # lsp and lspet dataset doesn't exist groundtruth of center points
        center_x = (lms[idx][0][lms[idx][0] < w].max() +
                    lms[idx][0][lms[idx][0] > 0].min()) / 2
        center_y = (lms[idx][1][lms[idx][1] < h].max() +
                    lms[idx][1][lms[idx][1] > 0].min()) / 2
        centers.append([center_x, center_y])

        scale = (lms[idx][1][lms[idx][1] < h].max() -
                lms[idx][1][lms[idx][1] > 0].min() + 4) / 368.0
        scales.append(scale)

    return kpts, centers, scales

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def getBoundingBox(img, kpt, height, width, stride):
    x = []
    y = []

    for index in range(0,len(kpt)):
        if float(kpt[index][1]) >= 0 or float(kpt[index][0]) >= 0:
            x.append(float(kpt[index][1]))
            y.append(float(kpt[index][0]))

    x_min = int(max(min(x), 0))
    x_max = int(min(max(x), width))
    y_min = int(max(min(y), 0))
    y_max = int(min(max(y), height))

    center_x = (x_min + x_max)/2
    center_y = (y_min + y_max)/2

    coord = []
    coord.append([min(int(center_y/stride),height/stride-1), min(int(center_x/stride),width/stride-1)])
    coord.append([min(int(y_min/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
    coord.append([min(int(y_min/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])
    coord.append([min(int(y_max/stride),height/stride-1),min(int(x_min/stride),width/stride-1)])
    coord.append([min(int(y_max/stride),height/stride-1),min(int(x_max/stride),width/stride-1)])

    box = np.zeros((int(height/stride), int(width/stride), 5), dtype=np.float32)
    for i in range(5):
        # resize from 368 to 46
        x = int(coord[i][0]) * 1.0
        y = int(coord[i][1]) * 1.0
        heat_map = guassian_kernel(size_h=int(height/stride), size_w=int(width/stride), center_x=x, center_y=y, sigma=3)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        box[:, :, i] = heat_map

    return box

class LSP_Dataset_Generator:

    def __init__(self, mode, root_dir, sigma, stride, transformer = None):

        self.img_list = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if ".jpg" in file]
        self.kpt_list, self.center_list, self.scale_list = read_mat_file(mode, root_dir, self.img_list)
        self.stride = stride
        self.transformer = transformer
        self.sigma = sigma
        self.bodyParts = [[13, 12], [12, 9], [12, 8], [8, 7], [9, 10], [7, 6], [10, 11], [12, 3], [2, 3], [2, 1], [1, 0], [3, 4], [4, 5]]

    def __getitem__(self, index):
        img_path = self.img_list[index]

        img = cv2.imread(img_path)
        kpt = self.kpt_list[index]
        center = self.center_list[index]
        scale = self.scale_list[index]

        # expand dataset
        if self.transformer:
            img, kpt, center = self.transformer(img, kpt, center, scale)

        height, width, _ = img.shape
        heatmap = np.zeros((round(height/self.stride), round(width/self.stride), int(len(kpt)+1)), dtype=np.float32)

        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height/self.stride+0.5),size_w=int(width/self.stride+0.5), center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map

        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  

        img = img.transpose([2, 0, 1])
        img = img / 255

        heatmap = heatmap.transpose([2, 0, 1])

        return img, heatmap

    def __len__(self):
        return len(self.img_list)


class LSPose:
    """LSPose Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Download data.')
            local_data_path = '/cache/data'
            sync_data(args.data_url, local_data_path, threads = 256)
            print('Create train and evaluate dataset.')

            train_dir = os.path.join(local_data_path, "LSP", "TRAIN")
            val_ir = os.path.join(local_data_path, "LSP" , "VAL")

            self.train_dataset = create_dataset_lsp(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_lsp(val_ir, training=False, args=args)
        else:
            train_dir = os.path.join(args.data_url, "TRAIN")
            val_ir = os.path.join(args.data_url, "VAL")

            if training:
                self.train_dataset = create_dataset_lsp(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_lsp(val_ir, training=False, args=args)


def create_dataset_lsp(dataset_dir, training, args):

    device_num, rank_id = _get_rank_info()
    shuffle = bool(training)

    if not training:
        data_set = ds.GeneratorDataset(LSP_Dataset_Generator('joints_test',
                                        dataset_dir, 
                                        args.sigma, 
                                        args.stride,
                                        transforms.Compose([
                                                transforms.TypeCast(),
                                                transforms.TestResized(size=(args.height, args.width)), ])
                                            ),
                                        column_names = ["img", "heatmap"],
                                        shuffle = shuffle,
                                        num_parallel_workers = args.num_parallel_workers)
                                      
    else:
        data_set = ds.GeneratorDataset(LSP_Dataset_Generator('joints_train',
                                    dataset_dir,
                                    args.sigma,
                                    args.stride,
                                    transforms.Compose([
                                        transforms.RandomColor(h_gain=0.8, s_gain=0.8, v_gain=0.8),
										transforms.GaussianBlur(kernel_size=7,prob=0.3,sigma=5),
                                        transforms.TypeCast(),
                                        transforms.RandomRotate(max_degree=10),
                                        transforms.TestResized(size=(args.height, args.width)), # H,W
                                        transforms.RandomHorizontalFlip()])
                                    ),
                                column_names=["img", "heatmap"],
                                shuffle= shuffle,
                                num_parallel_workers = args.num_parallel_workers,
                                num_shards = device_num,
                                shard_id = rank_id)


    data_set = data_set.batch(args.batch_size, drop_remainder= False,
                            num_parallel_workers=args.num_parallel_workers)

    return data_set


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id