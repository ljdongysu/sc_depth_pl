import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import torch
from scipy import sparse


def load_sparse_depth(filename):
    sparse_depth = sparse.load_npz(filename)
    depth = sparse_depth.todense()
    return np.array(depth)


def crawl_folders(folders_list, dataset='nyu'):
    imgs = []
    depths = []
    for folder in folders_list:
        current_imgs = sorted(folder.files('*.jpg'))
        if dataset in ['nyu', 'bonn', 'tum']:
            current_depth = sorted((folder/'depth/').files('*.png'))
        elif dataset in ['ddad', 'kitti']:
            current_depth = sorted((folder/'depth/').files('*.npz'))
        imgs.extend(current_imgs)
        depths.extend(current_depth)
    return imgs, depths


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, transform=None, dataset='nyu'):
        self.root = Path(root)/'training'
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1]
                       for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.imgs, self.depth = crawl_folders(self.scenes, self.dataset)

    def __getitem__(self, index):
        img = imread(self.imgs[index]).astype(np.float32)

        if self.dataset in ['nyu']:
            depth = torch.from_numpy(
                imread(self.depth[index]).astype(np.float32)).float()/5000
        if self.dataset in ['bonn', 'tum']:
            depth = torch.from_numpy(
                imread(self.depth[index]).astype(np.float32)).float()/1000
        elif self.dataset in ['kitti', 'ddad']:
            depth = torch.from_numpy(load_sparse_depth(
                self.depth[index]).astype(np.float32))

        if self.transform is not None:
            img, _ = self.transform([img], None)
            img = img[0]

        return img, depth

    def __len__(self):
        return len(self.imgs)
