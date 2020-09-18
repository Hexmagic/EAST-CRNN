import os
import sys
from argparse import ArgumentParser

import torch
from ignite.engine import Events, create_supervised_trainer, Engine
from ignite.metrics import Loss
from torch.utils import data

from dataset.east import custom_dataset
from model.east import EAST, EASTLoss


def create_supervised_trainer(model, optimizer, criterion, device=None):

    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        img, gt_score, gt_geo, ignored_map = batch
        img, gt_score, gt_geo, ignored_map = img.to(device), gt_score.to(
            device), gt_geo.to(device), ignored_map.to(device)
        pred_score, pred_geo = model(img)
        loss = criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    return Engine(_update)

from dataset.crnn import ResizeNormalize
if __name__ == '__main__':
    parser = ArgumentParser('EAST Train')
    parser.add_argument('--train_img',
                        type=str,
                        default='data/img',
                        help='train image folder')
    parser.add_argument('--train_gt',
                        type=str,
                        default='data/gt',
                        help='train gt folder')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch size')
    parser.add_argument('--n_cpu',
                        type=int,
                        default=8,
                        help='num workers used to handler dataloader')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='model learning rate')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--pretrained',
                        type=str,
                        default='',
                        help='pretrained model')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='max train epochs')
    parser.add_argument('--save_folder',
                        type=str,
                        default='pths',
                        help='model save path')
    parser.add_argument('--save_interval',
                        type=int,
                        default=5,
                        help='save model every save_interval')
    parser.add_argument('--start_iter', type=int, default=0, help='start iter')
    #parser.add_argument('--pths', type=str, default='pths', help='weight store folder')
    opt = parser.parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    model = EAST()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    trainset = custom_dataset(opt.train_img, opt.train_gt)
    train_loader = data.DataLoader(trainset, batch_size=opt.batch_size, \
                                        shuffle=True, num_workers=opt.n_cpu, drop_last=True)
    if opt.pretrained:
        model.load_state_dict(torch.load(opt.pretrained))
    trainer = create_supervised_trainer(model, optimizer, EASTLoss(), device)
    if not os.path.exists('pths'):
        os.mkdir('pths')
    if not os.path.exists('pths/vgg16_bn-6c64b313.pth'):
        print("model vgg16_bn-6c64b313.pth not Exists!!!")
        sys.exit(0)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        if engine.state.iteration % opt.log_interval == 0:
            print("Epoch{}[{}/{}] Loss: {:.2f}".format(
                engine.state.epoch,
                engine.state.iteration % (len(train_loader)),
                len(train_loader), engine.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine):
        if (engine.state.epoch + 1) % opt.save_interval == 0:
            state_dict = model.state_dict()
            torch.save(
                state_dict,
                os.path.join(opt.save_folder,
                             'east_{}.pth'.format(engine.state.epoch + 1)))

    trainer.run(train_loader, max_epochs=opt.epochs)
