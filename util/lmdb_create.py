import argparse
import os
import shutil
import sys

from PIL import Image

import lmdb


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        labelPath = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        img = Image.open(imagePath)
        imageBin = img.tobytes()
        with open(labelPath, 'r') as f:
            for line in f:
                lst = line.strip().split(',')
                key = lst[-1]
                a, b, c, d, e, f, g, h = key
                crop = img.crop([a, b, g, h])
                imageKey = 'image-%09d' % cnt
                labelKey = 'label-%09d' % cnt
                cache[imageKey] = crop.tobytes()
                cache[labelKey] = key
                cnt += 1
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def read_data_from_folder(root):
    image_path_list = []
    label_list = []
    pics = os.listdir(f'{root}/img')
    pics.sort(key=lambda i: len(i))
    for pic in pics:
        img_path = '{root}/img/{pic}'
        img_id = pic.split('.')[0]
        label_path = f'{root}/gt/gt_{img_id}.txt'
        image_path_list.append(img_path)
        label_list.append(label_path)
    return image_path_list, label_list


def show_demo(demo_number, image_path_list, label_list):
    print('\nShow some demo to prevent creating wrong lmdb data')
    print(
        'The first line is the path to image and the second line is the image label'
    )
    for i in range(demo_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='lmdb data output path')
    parser.add_argument('--folder',
                        type=str,
                        help='path to folder which contains the images')
    args = parser.parse_args()

from sklearn.model_selection import train_test_split


def create_lmdb(root, train_out, val_out):
    image_path_list, label_list = read_data_from_folder(root)
    train_img, val_img, train_label, val_label = train_test_split(
        image_path_list, label_list)
    print("创建训练数据集")
    createDataset(train_out, train_img, train_label)
    print("创建测试数据集")
    createDataset(val_out, val_img, val_label)
    #show_demo(2, image_path_list, label_list)
