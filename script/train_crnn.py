from __future__ import division, print_function

import argparse
from operator import mod
import os
import pdb

import torch
import torch.optim as optim
import torch.utils.data
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss
from scipy._lib.decorator import decorate
from torch.autograd import Variable
from torch.nn import CTCLoss

from config.crnn_cfg import lmdb_train_path, lmdb_val_path
from dataset.crnn import AlignCollate, LmdbDataset, ResizeNormalize
from model.crnn import CRNN
from util.label import LabelDecoder, LabelEncoder
from util.lmdb_create import create_lmdb

torch.multiprocessing.set_sharing_strategy('file_system')

encoder = LabelEncoder()
decoder = LabelDecoder()
parser = argparse.ArgumentParser()
parser.add_argument("-train",
                    "--trainroot",
                    default=lmdb_train_path,
                    help="path to train dataset")
parser.add_argument("-val",
                    "--valroot",
                    default=lmdb_val_path,
                    help="path to val dataset")
parser.add_argument("-H", type=int, default=32)
parser.add_argument("-W", type=int, default=100)
parser.add_argument('-root', type=str, default='data')
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-n_cpu", type=int, default=8)
parser.add_argument("-epochs", type=int, default=20)
parser.add_argument("-nc", type=int, default=1)
parser.add_argument("-nclass", type=int, default=24)
parser.add_argument("-nh", type=int, default=256)
parser.add_argument("-pretrained", type=str, default="")
parser.add_argument("-expr_dir", default="pths")
parser.add_argument("-lr", type=float, default=1e-4)
parser.add_argument("-log_interval", type=int, default=50)
parser.add_argument("-save_interval", type=int, default=5)
args = parser.parse_args()
print(args)


def create_supervised_trainer(model, optimizer, criterion, device=None):
    model.to(device)

    def _update(engine, batch):
        img, texts, length = batch
        batch_size = img.size(0)
        img = Variable(img).cuda()
        optimizer.zero_grad()
        preds = model(img)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))

        #ctc_loss() received an invalid combination of arguments - got (Tensor, tuple, Tensor, tuple, int, int, bool),
        loss = criterion(preds, texts, tuple(preds_size), length) / batch_size
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)


def valid(model, val_loader):
    with torch.no_grad():
        n_correct = 0
        model.eval()
        length = len(val_loader)
        for batch in val_loader:
            img, texts = batch
            batch_size = img.size(0)
            img = Variable(img).cuda()
            preds = model(img)
            #loss = criterion(preds, texts, preds_size, length) / batch_size
            preds_size = Variable(
                torch.LongTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = decoder.decode(preds.data, preds_size.data, raw=False)
            cpu_texts_decode = []
            for i in texts:
                cpu_texts_decode.append(i.decode('utf-8', 'strict'))
            for pred, target in zip(sim_preds, cpu_texts_decode):
                if pred == target:
                    n_correct += 1
            #return preds, texts
        print(f'Acc :{n_correct/float(length*args.batch_size)}')
    model.train()


def get_data_loader(args):
    # train
    train_dataset = LmdbDataset(root=args.trainroot)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=int(args.n_cpu),
                                               collate_fn=AlignCollate(
                                                   h=args.H, w=args.W),
                                               drop_last=True)

    val_dataset = LmdbDataset(root=args.valroot,
                              transform=ResizeNormalize(args.H, args.W),
                              target_transform=None)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             shuffle=True,
                                             batch_size=64,
                                             num_workers=int(args.n_cpu),
                                             drop_last=True)
    return train_loader, val_loader


def main():
    device = torch.device('cuda')
    model = CRNN(args.nc, args.nclass, args.nh)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = CTCLoss(zero_infinity=True).cuda()
    if not os.path.exists(lmdb_train_path):
        create_lmdb(args.root, args.trainroot, args.valroot)
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

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine: Engine):
        if engine.state.iteration % args.log_interval == 0:
            print("Epoch {} [{}/{}] :Loss {}".format(
                engine.state.epoch,
                engine.state.iteration % (len(train_loader)),
                len(train_loader), engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_acc(engine):
        valid(model, val_loader)
        if engine.state.epoch % args.save_interval == 0:
            torch.save(model, f'{args.expr_dir}/crnn_{engine.state.epoch}.pth')

    trainer.run(train_loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
