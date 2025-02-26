import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import random
import os

CONFIG_FILE = ['config.yaml', 'MODULE.yaml', 'MoudleParam.yaml']

def load_as_float(path):
    return imread(path).astype(np.float32)


def generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length-1)//2
    shifts = list(range(-demi_length * k,
                        demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames-demi_length * k):
            sample_index = {'tgt_idx': i, 'ref_idx': []}
            for j in shifts:
                sample_index['ref_idx'].append(i+j)
            sample_index_list.append(sample_index)

    return sample_index_list


class TrainFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self,
                 root,
                 train=True,
                 sequence_length=3,
                 transform=None,
                 skip_frames=1,
                 dataset='kitti',
                 use_frame_index=False,
                 with_pseudo_depth=False,
                 file_list=None,
                 depth_dir='DEPTH/AdelaiDepth'):
        np.random.seed(0)
        random.seed(0)
        self.dataset = dataset
        self.k = skip_frames
        self.with_pseudo_depth = with_pseudo_depth
        self.transform = transform
        self.depth_dir = depth_dir
        if self.dataset == 'indemind':
            self.root = root
            self.K_dict = {}
            self.K = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]], dtype=np.float32)
            self.file_list = file_list
            self.read_indemind_data(self.root, self.file_list)
        else:
            self.root = Path(root)/'training'
            scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
            self.scenes = [self.root/folder[:-1]
                           for folder in open(scene_list_path)]
            self.use_frame_index = use_frame_index
            self.crawl_folders(sequence_length)

    def GetConfigFile(self, path):
        for file_name in CONFIG_FILE:
            file = os.path.join(path, file_name)
            if os.path.exists(file):
                break
        return file

    def set_by_config_yaml(self, folder):
        image_path_lehgth = len(folder.split('/'))
        for index in range(1, image_path_lehgth):
            config_file = "/" + os.path.join(*(folder.split('/')[:-1 * index]))
            config_file = self.GetConfigFile(config_file)
            if os.path.exists(config_file):
                break
        if config_file in self.K_dict:
            return self.K_dict[config_file]
        else:
            with open(config_file, 'r') as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if "Pl" in lines[i]:
                        config_Pl_x = lines[i + 4]
                        Pl_00 = config_Pl_x.split(' ')[5]
                        Pl_02 = config_Pl_x.split(' ')[7]
                        config_Pl_y = lines[i + 5]

                        Pl_11 = config_Pl_y.split(' ')[7]
                        Pl_12 = config_Pl_y.split(' ')[8]
                        self.K[0][0] = float(Pl_00.split(',')[0])
                        self.K[0][2] = float(Pl_02.split(',')[0])
                        self.K[1][1] = float(Pl_11.split(',')[0])
                        self.K[1][2] = float(Pl_12.split(',')[0])
                        self.K_dict[config_file] = self.K
                        return self.K

    def read_indemind_data(self, root, file_list):
        sequence_set = []
        with open(file_list,'r') as f:
            frames_list = f.readlines()
            for frame_list in frames_list:
                frame_before, frame_current, frame_after, _ = frame_list.split()
                sample = {'tgt_img': Path(os.path.join(root,frame_current))}
                sample['ref_imgs'] = []
                sample['ref_imgs'].append(Path(os.path.join(root,frame_before)))
                sample['ref_imgs'].append(Path(os.path.join(root,frame_after)))
                frame_current_png = os.path.splitext(frame_current)[0] + '.png'
                sample['tgt_pseudo_depth'] = Path(os.path.join(self.depth_dir, frame_current_png))
                sample['intrinsics'] = self.set_by_config_yaml(os.path.join(root,frame_current))
                sequence_set.append(sample)

        self.samples = sequence_set

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []

        for scene in self.scenes:
            intrinsics = np.genfromtxt(
                scene/'cam.txt').astype(np.float32).reshape((3, 3))

            imgs = sorted(scene.files('*.jpg'))

            if self.use_frame_index:
                frame_index = [int(index)
                               for index in open(scene/'frame_index.txt')]
                imgs = [imgs[d] for d in frame_index]

            if self.with_pseudo_depth:
                pseudo_depths = sorted((scene/'leres_depth').files('*.png'))
                if self.use_frame_index:
                    pseudo_depths = [pseudo_depths[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            sample_index_list = generate_sample_index(
                len(imgs), self.k, sequence_length)
            for sample_index in sample_index_list:
                sample = {'intrinsics': intrinsics,
                          'tgt_img': imgs[sample_index['tgt_idx']]}
                if self.with_pseudo_depth:
                    sample['tgt_pseudo_depth'] = pseudo_depths[sample_index['tgt_idx']]

                sample['ref_imgs'] = []
                for j in sample_index['ref_idx']:
                    sample['ref_imgs'].append(imgs[j])
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt_img'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.with_pseudo_depth:
            tgt_pseudo_depth = load_as_float(sample['tgt_pseudo_depth'])

        if self.transform is not None:
            if self.with_pseudo_depth:
                imgs, intrinsics = self.transform(
                    [tgt_img, tgt_pseudo_depth] + ref_imgs, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                tgt_pseudo_depth = imgs[1]
                ref_imgs = imgs[2:]
            else:
                imgs, intrinsics = self.transform(
                    [tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
                tgt_img = imgs[0]
                ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])

        if self.with_pseudo_depth:
            return tgt_img, tgt_pseudo_depth, ref_imgs, intrinsics
        else:
            return tgt_img, ref_imgs, intrinsics

    def __len__(self):
        return len(self.samples)
