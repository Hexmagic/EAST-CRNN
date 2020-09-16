from __future__ import division, print_function

import argparse
import os
from argparse import Namespace
import pdb

import numpy as np
import torch
import random
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.nn import CTCLoss
from tqdm import tqdm

import models.crnn as net
from data import dataset


class Trainer(object):
    def __init__(self, args: Namespace) -> None:
        self.loss_list = []
        self.model = net.CRNN(args.nc, args.nclass, args.nh).cuda()
        if args.pretrained:
            self.model.load_state_dict(torch.load(args.pretrained))
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.criterion = CTCLoss(zero_infinity=True).cuda()
        self.args = args

    def run(self):
        for epoch in tqdm(range(self.args.epochs)):
            train_loader, val_loader = self.get_data_loader()
            train_bar = tqdm(train_loader)
            val_bar = tqdm(val_loader)
            train_bar.set_postfix_str(f"[Train {epoch}/{self.args.epochs}]")
            val_bar.set_postfix_str(f"[Val {epoch}/{self.args.epochs}]")
            self.train(train_bar)
            if epoch % 5 == 0:
                self.val(val_bar)
                torch.save(
                    self.model.state_dict(),
                    "{0}/netCRNN_{1}.pth".format(self.args.expr_dir, epoch),
                )

    def get_data_loader(self):
        # train
        train_dataset = dataset.lmdbDataset(root=self.args.trainroot)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=int(self.args.n_cpu),
            collate_fn=dataset.alignCollate(h=self.args.H, w=self.args.W),
        )

        val_dataset = dataset.lmdbDataset(
            root=self.args.valroot,
            transform=dataset.resizeNormalize(h=self.args.H, w=self.args.W),
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            num_workers=int(self.args.n_cpu),
        )
        return train_loader, val_loader

    def val(self, val_bar):
        with torch.no_grad():
            self.model.eval()
            i = 0
            n_correct = 0
            val_loss = []
            for batch in range(val_bar):
                img, texts, length = batch
                batch_size = img.size(0)
                img, texts, length = Variable(img), Variable(texts), Variable(length)
                img, texts, length = img.cuda(), texts.cuda(), length.cuda()
                preds = self.model(img)
                preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
                loss = self.criterion(preds, texts, preds_size, length) / batch_size
                val_loss.append(loss.item(0))
                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                # sim_preds = converter.decode(preds.data,
                #                              preds_size.data,
                #                              raw=False)
                # cpu_texts_decode = []
                # for i in cpu_texts:
                #     cpu_texts_decode.append(i.decode("utf-8", "strict"))
                # for pred, target in zip(sim_preds, cpu_texts_decode):
                #     if pred == target:
                #         n_correct += 1

            accuracy = n_correct / float(max_iter * params.batchSize)
            print("Val loss: %f, accuray: %f" % (np.mean(val_loss), accuracy))

    def train(self, train_bar: tqdm):
        self.model.train()
        i = 0
        for batch in train_bar:
            i += 1
            img, texts, length = batch
            batch_size = img.size(0)
            img = Variable(img).cuda()
            self.optimizer.zero_grad()
            preds = self.model(img)
            preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
            loss = self.criterion(preds, texts, preds_size, length) / batch_size
            self.loss_list.append(random.random())
            if i % 500 == 0:
                train_bar.set_description_str(f"Loss {np.mean(self.loss_list)}")
            #loss.backward()
            self.optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train", "--trainroot", required=True, help="path to train dataset"
    )
    parser.add_argument("-val", "--valroot", required=True, help="path to val dataset")
    parser.add_argument("-H", type=int, default=32)
    parser.add_argument("-W", type=int, default=100)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-n_cpu", type=int, default=8)
    parser.add_argument("-epochs", type=int, default=1000)
    parser.add_argument("-nc", type=int, default=1)
    parser.add_argument("-nclass", type=int, default=26)
    parser.add_argument("-nh", type=int, default=256)
    parser.add_argument("-pretrained", type=str, default="")
    parser.add_argument("-expr_dir", default="expr")
    parser.add_argument("-lr", type=float, default=1e-4)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.expr_dir):
        os.makedirs(args.expr_dir)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
