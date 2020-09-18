import argparse
import json
import os
#from pytesseract import image_to_string
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import lanms
import numpy as np
from numpy.lib.type_check import imag
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from dataset.crnn import ResizeNormalize
from dataset.east import get_rotate_mat
from model.east import EAST
from util.label import LabelDecoder

decoder = LabelDecoder()
parser = argparse.ArgumentParser('EAST Detect')
parser.add_argument('--folder',
                    type=str,
                    default='sample',
                    help='detect imgs folder')
parser.add_argument('--n_cpu', type=int, default=1)
parser.add_argument('--output',
                    type=str,
                    default='output',
                    help='output folder ,default is output')
parser.add_argument('--east',
                    type=str,
                    default='pths/east_50.pth',
                    help='pretrained east path')
parser.add_argument('--crnn',
                    type=str,
                    default='pths/crnn_20.pth',
                    help='pretrained crnn model')
args = parser.parse_args()
print(args)


class Demo(object):
    def __init__(self) -> None:
        self.crnn = torch.load(args.crnn).cuda()
        self.crnn.eval()
        self.east = EAST().cuda()
        self.east.load_state_dict(torch.load(args.east))
        self.east.eval()
        self.transform = ResizeNormalize(32, 100)

    def resize_img(self, img):
        '''resize image to be divisible by 32
        '''
        w, h = img.size
        resize_w = w
        resize_h = h

        resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
        resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
        img = img.resize((resize_w, resize_h), Image.BILINEAR)
        ratio_h = resize_h / h
        ratio_w = resize_w / w

        return img, ratio_h, ratio_w

    def load_pil(self, img):
        '''convert PIL Image to torch.Tensor
        '''
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return t(img).unsqueeze(0)

    def detect_text(self, img):
        image = Variable(self.transform(img)).cuda().unsqueeze(0)
        preds = self.crnn(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([26] * 1))
        # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = decoder.decode(preds.data, preds_size.data, raw=False)
        return sim_pred

    def is_valid_poly(self, res, score_shape, scale):
        '''check if the poly in image scope
        Input:
            res        : restored poly in original image
            score_shape: score map shape
            scale      : feature map -> image
        Output:
            True if valid
        '''
        cnt = 0
        for i in range(res.shape[1]):
            if res[0,i] < 0 or res[0,i] >= score_shape[1] * scale or \
            res[1,i] < 0 or res[1,i] >= score_shape[0] * scale:
                cnt += 1
        return True if cnt <= 1 else False

    def restore_polys(self, valid_pos, valid_geo, score_shape, scale=4):
        '''restore polys from feature maps in given positions
        Input:
            valid_pos  : potential text positions <numpy.ndarray, (n,2)>
            valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
            score_shape: shape of score map
            scale      : image / feature map
        Output:
            restored polys <numpy.ndarray, (n,8)>, index
        '''
        polys = []
        index = []
        valid_pos *= scale
        d = valid_geo[:4, :]  # 4 x N
        angle = valid_geo[4, :]  # N,

        for i in range(valid_pos.shape[0]):
            x = valid_pos[i, 0]
            y = valid_pos[i, 1]
            y_min = y - d[0, i]
            y_max = y + d[1, i]
            x_min = x - d[2, i]
            x_max = x + d[3, i]
            rotate_mat = get_rotate_mat(-angle[i])

            temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
            temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
            coordidates = np.concatenate((temp_x, temp_y), axis=0)
            res = np.dot(rotate_mat, coordidates)
            res[0, :] += x
            res[1, :] += y

            if self.is_valid_poly(res, score_shape, scale):
                index.append(i)
                polys.append([
                    res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2],
                    res[1, 2], res[0, 3], res[1, 3]
                ])
        return np.array(polys), index

    def get_boxes(self, score, geo, score_thresh=0.9, nms_thresh=0.2):
        '''get boxes from feature map
        Input:
            score       : score map from model <numpy.ndarray, (1,row,col)>
            geo         : geo map from model <numpy.ndarray, (5,row,col)>
            score_thresh: threshold to segment score map
            nms_thresh  : threshold in nms
        Output:
            boxes       : final polys <numpy.ndarray, (n,9)>
        '''
        score = score[0, :, :]
        xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c]
        if xy_text.size == 0:
            return None

        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]
        valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n
        polys_restored, index = self.restore_polys(valid_pos, valid_geo,
                                                   score.shape)
        if polys_restored.size == 0:
            return None

        boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = polys_restored
        boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
        return boxes

    def adjust_ratio(self, boxes, ratio_w, ratio_h):
        '''refine boxes
        Input:
            boxes  : detected polys <numpy.ndarray, (n,9)>
            ratio_w: ratio of width
            ratio_h: ratio of height
        Output:
            refined boxes
        '''
        if boxes is None or boxes.size == 0:
            return []
        boxes[:, [0, 2, 4, 6]] /= ratio_w
        boxes[:, [1, 3, 5, 7]] /= ratio_h
        return np.around(boxes)

    def detect(self, img, device):
        '''detect text regions of img using model
        Input:
            img   : PIL Image
            model : detection model
            device: gpu if gpu is available
        Output:
            detected polys
        '''
        img, ratio_h, ratio_w = self.resize_img(img)
        with torch.no_grad():
            score, geo = self.east(self.load_pil(img).to(device))
        boxes = self.get_boxes(
            score.squeeze(0).cpu().numpy(),
            geo.squeeze(0).cpu().numpy())
        return self.adjust_ratio(boxes, ratio_w, ratio_h)

    def plot_boxes(self, img, boxes):
        '''plot boxes on image
        '''
        if boxes is None:
            return img

        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.polygon([
                box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]
            ],
                         outline=(0, 255, 0))
        return img

    def detect_boxes(self, paths):
        rtn = {}

        for path in tqdm(paths, desc=f"Detector "):
            arr = cv2.imread(path)
            img = Image.open(path)
            #img = new(img)
            img = img.convert('RGB')
            img_name = path.split('/')[-1]
            boxes = self.detect(img, 'cuda:0')
            #img.save(f'output/{img_name}')
            img = self.infer(boxes, img)
            img = self.plot_boxes(img, boxes)
            img.save(f'{args.output}/{img_name}')

    def infer(self, boxes, img):
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('Hack-Bold.ttf', size=30)
        for box in boxes:
            x, y = int(box[0]), int(box[1])
            rx, ry = int(box[4]), int(box[5])
            crop = img.crop((x, y, rx, ry)).convert('L')
            text = self.detect_text(crop)
            draw.text((x, y - 30), text, (128, 0, 128), font)
        return img


if __name__ == '__main__':
    demo = Demo()
    if args.folder:
        names = [
            os.path.join(args.folder, ele) for ele in os.listdir(args.folder)
        ]
        demo.detect_boxes(names)
    else:
        print("FOlder or Image Should be Provided")
