from __future__ import division, print_function

import argparse
from operator import mod
import os
import random
from argparse import Namespace
from sys import ps1

import numpy as np
import torch
from torch._C import device
import torch.optim as optim
import torch.utils.data
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import Accuracy, Loss
from torch.autograd import Variable
from torch.nn import CTCLoss
from tqdm import tqdm

import dataset.crnn as dataset
from model.crnn import CRNN


def create_supervised_trainer(model, optimizer, criterion, device=None):
    model.to(device)

    def _update(engine, batch):
        img, texts, length = batch
        batch_size = img.size(0)
        img = Variable(img).cuda()
        optimizer.zero_grad()
        preds = model(img)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        loss = criterion(preds, texts, preds_size, length) / batch_size

        optimizer.step()
        return loss.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device, output_transform=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        with torch.no_grad():
            img, texts, length = batch
            batch_size = img.size(0)
            img, texts, length = Variable(img), Variable(texts), Variable(
                length)
            img, texts, length = img.cuda(), texts.cuda(), length.cuda()
            preds = model(img)
            preds_size = Variable(
                torch.LongTensor([preds.size(0)] * batch_size))
            #loss = criterion(preds, texts, preds_size, length) / batch_size
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            return output_transform(texts, preds)

    evaluator = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    return evaluator


def get_data_loader(args):
    # train
    train_dataset = dataset.lmdbDataset(root=args.trainroot)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.n_cpu),
        collate_fn=dataset.alignCollate(h=args.H, w=args.W),
    )

    val_dataset = dataset.lmdbDataset(
        root=args.valroot,
        transform=dataset.resizeNormalize(h=args.H, w=args.W),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=int(args.n_cpu),
    )
    return train_loader, val_loader


def main():
    from config.crnn_cfg import lmdb_train_path, lmdb_val_path
    from util.lmdb_create import create_lmdb

    parser = argparse.ArgumentParser()
    parser.add_argument("-train",
                        "--trainroot",
                        default=lmdb_train_path,
                        required=True,
                        help="path to train dataset")
    parser.add_argument("-val",
                        "--valroot",
                        default=lmdb_val_path,
                        required=True,
                        help="path to val dataset")
    parser.add_argument("-H", type=int, default=32)
    parser.add_argument("-W", type=int, default=100)
    parser.add_argument('-root', type=str, default='data')
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
    device = torch.device('cuda')
    model = CRNN(args.nc, args.nclass, args.nh)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = CTCLoss(zero_infinity=True).cuda()
    if not os.path.exists(lmdb_train_path):
        create_lmdb(args.root, args.train, args.val)
    if not os.path.exists(args.expr_dir):
        os.makedirs(args.expr_dir)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    train_loader, val_loader = get_data_loader(args)
    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        criterion,
                                        device=device)
    evaluator = create_supervised_evaluator(model, {},
                                            device,
                                            output_transform=None)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        pass

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_acc(engine):
        evaluator.run(val_loader)
        metric = evaluator.state.metrics
        print(metric)

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
