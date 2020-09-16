#!/usr/bin/python
# encoding: utf-8

import torch
from torch.utils.data import Dataset
from tool.label import LabelEncoder
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image


class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=LabelEncoder()):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            print("cannot creat lmdb from %s" % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get("num-samples".encode("utf-8")))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = "image-%09d" % index
            imgbuf = txn.get(img_key.encode("utf-8"))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            img = Image.open(buf).convert("L")
            if self.transform is not None:
                img = self.transform(img)
            label_key = "label-%09d" % index
            label = txn.get(label_key.encode("utf-8"))

            if self.target_transform is not None:
                text, length = self.target_transform(label)
                return (img, text, length)

        return (img, label)


class resizeNormalize(object):
    def __init__(self, h, w, interpolation=Image.BILINEAR):
        self.size = (w, h)
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):
    def __init__(self, h, w):
        self.imgH = h
        self.imgW = w

    def __call__(self, batch):
        images, labels, lengths = zip(*batch)
        transform = resizeNormalize(self.imgH, self.imgW)
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)
        #$labels = torch.stack(labels, dim=1)
        return images, labels, lengths


if __name__ == "__main__":
    data = lmdbDataset(root="train_out")
    print(data[0])
