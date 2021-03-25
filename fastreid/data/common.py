# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel: pid = self.pid_dict[pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path
        }

    @property
    def num_classes(self):
        return len(self.pids)

class PairDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, hazy_img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.hazy_img_items = hazy_img_items
        self.transform = transform
        self.relabel = relabel

        self.pid_dict = {}
        if self.relabel:
            pids = list()
            for i, item in enumerate(img_items):
                if item[1] in pids: continue
                pids.append(item[1])
            self.pids = pids
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        hazy_img_path, hazy_pid, hazy_camid = self.hazy_img_items[index]
        img = read_image(img_path)
        hazy_img = read_image(hazy_img_path)
        if self.transform is not None: img = self.transform(img)
        if self.transform is not None: hazy_img = self.transform(hazy_img)
        if self.relabel: pid = self.pid_dict[pid]
        if self.relabel: hazy_pid = self.pid_dict[hazy_pid]
        return {
            "images": img,
            "targets": pid,
            "camid": camid,
            "img_path": img_path,
            "hazy_images": hazy_img,
            "hazy_targets": hazy_pid,
            "hazy_camid": hazy_camid,
            "hazy_img_path": hazy_img_path
        }

    @property
    def num_classes(self):
        return len(self.pids)
