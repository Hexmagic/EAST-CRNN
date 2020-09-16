import json
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import tool.utils as utils
from data import dataset
from PIL import Image
import os
import models.crnn as crnn
import params
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=False,
    default="expr/netCRNN_151_1000.pth",
    help="crnn model path",
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    default="val",
    required=False,
    help="demo image path",
)
args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path

# net init
nclass = len(params.alphabet) + 1
model = crnn.CRNN(params.imgH, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(params.alphabet)


class DemoData(Dataset):
    def __init__(self, root):
        super(DemoData, self).__init__()
        self.root = root
        self.files = os.listdir(self.root)
        self.transform = dataset.resizeNormalize((100, 32))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = f"{self.root}/{self.files[index]}"
        image = Image.open(path).convert("L")
        image = self.transform(image)
        return self.files[index], image


def main():
    model.eval()
    dataset = DemoData(root=args.image_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    rst = {}
    i = 0
    import shutil
    for batch in tqdm(loader):
        i += 1
        paths, image = batch
        image = image.cuda()
        size = args.batch_size
        image = image.view(size, *image.size()[1:])
        """ image = Image.open("H_8.png").convert("L")
        image = dataset.transform(image)
        image = image.cuda()
        image = image.view(1, *image.size()) """
        image = Variable(image)
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = Variable(torch.LongTensor([26] * size))
        # raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        # print("%-20s => %-20s" % (raw_pred, sim_pred))
        j = 0
        for p, d in zip(paths, sim_pred):
            shutil.copyfile(f'{args.image_path}/{p}', f'test/{p}_{d}.png')
            #rst[f'{i}_{j}_{p}'] = d
    with open("rst.json", "w") as f:
        json.dump(rst, f)


if __name__ == "__main__":
    main()
