import argparse
import logging
import os
import sys
from os import path
from sys import exit

import datetime as dt
import time

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np

import torch
th = torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from core.dataset import PNGDataset
from core.training_criteria import soft_dice_loss, accuracy, l1_loss_sum, l1_loss_mean, l2_loss_sum, l2_loss_mean


def timestamp():
    return dt.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

class ModelHelper():
    def __init__(self, model_gen, model_disc, opter_gen, opter_disc, disc_otp_shape=tuple([1])):
        self.model_gen, self.model_disc = model_gen, model_disc
        self.opter_gen, self.opter_disc = opter_gen, opter_disc
        self.disc_otp_shape = disc_otp_shape

    def _disc_otp_shape(self, x):
        return tuple([x.shape[0]]) + self.disc_otp_shape

    def _toggle_model_params_grads(self, model, requires_grad=False):
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad

    def generator_arguments(self, x, y):
        x_ = x.detach()
        self._toggle_model_params_grads(self.model_disc, False)
        self._toggle_model_params_grads(self.model_gen, True)
        self.model_gen.train()
        self.model_disc.eval()

        y_gen_fake = self.model_gen(x_)  #y.detach()
        x_disc = th.cat((x_, y_gen_fake), dim=1)
        y_disc_fake = self.model_disc(x_disc)
        return x_, y_gen_fake, [y_disc_fake, th.ones(self._disc_otp_shape(x_))]

    def discriminator_arguments(self, x, y, i_batch):
        x_ = x.detach()
        use_fake = i_batch % 2 == 0
        self._toggle_model_params_grads(self.model_gen, False)
        self._toggle_model_params_grads(self.model_disc, True)
        self.model_gen.eval()
        self.model_disc.train()

        if use_fake:
            y_gen_fake = self.model_gen(x_)
            x_disc_fake = th.cat((x_, y_gen_fake), dim=1)
            y_disc_fake = self.model_disc(x_disc_fake)
            return x_, y_gen_fake, [y_disc_fake, th.zeros(self._disc_otp_shape(x_))]
        else:
            y_ = y.detach()
            x_disc_true = th.cat((x_, y_), dim=1)
            y_disc_true = self.model_disc(x_disc_true)
            return x_, y_, [y_disc_true, th.ones(self._disc_otp_shape(x_))]

weighted_losses = lambda losses_funcs, losses_args, losses_weights: [_func(*_args) * _wgt for _func, _args, _wgt in zip(losses_funcs, losses_args, losses_weights)]

class ModelEvaluator():
    def __init__(self, model_gen, model_disc, losses_gen_funcs, loss_disc_func, device=None):
        self.model_disc = model_disc
        self.model_gen = model_gen
        self.loss_disc_func = loss_disc_func
        self.losses_gen_funcs = losses_gen_funcs
        self.device = device

    def _weighted_losses(self, losses_funcs, losses_args, losses_weights):
        return [_func(*_args) * _wgt for _func, _args, _wgt in zip(losses_funcs, losses_weights, losses_args)]

    def __call__(self, loader):
        self.model_gen.eval()
        self.model_disc.eval()

        type_y = torch.float32 #if net.n_classes == 1 else torch.long
        n_val = len(loader)  # the number of batch
        tot = 0

        self.y_fake = []
        self.y_true = []
        self.x = []
        self.losses_gen = []
        self.losses_disc = []
        self.accs_disc = []

        with tqdm(total=n_val, desc='Validation Round', unit='batch', leave=False) as pbar:
            for batch in loader:
                x = batch['x'].to(device=self.device, dtype=torch.float32)
                y_true = batch['y'].to(device=self.device, dtype=torch.float32).unsqueeze(1)
                y_fake = self.model_gen(x)
                x_disc_fake = th.cat((x, y_fake), dim=1)
                x_disc_true = th.cat((x, y_true), dim=1)
                y_disc_on_fake = self.model_disc(x_disc_fake)
                y_disc_on_true = self.model_disc(x_disc_true)
                y_disc_on_fake_true = th.zeros(y_disc_on_fake.shape)
                y_disc_on_true_true = th.ones(y_disc_on_true.shape)
                y_disc_gen_trgts = th.ones(y_disc_on_true.shape)

                losses_gen_args = [[y_fake, y_true], [y_disc_on_fake, y_disc_gen_trgts]]

                with torch.no_grad():
                    losses_disc = [self.loss_disc_func(y_disc_on_fake, y_disc_on_fake_true).item(),
                                   self.loss_disc_func(y_disc_on_true, y_disc_on_true_true).item()]
                    accs_disc = [accuracy(y_disc_on_fake, y_disc_on_fake_true).item(),
                                 accuracy(y_disc_on_true, y_disc_on_true_true).item()]
                    losses_gen = [_l.item() for _l in self._weighted_losses(self.losses_gen_funcs[0], self.losses_gen_funcs[1], losses_gen_args)]

                self.x.append(x)
                self.y_true.append(y_true)
                self.y_fake.append(y_fake)
                self.losses_gen.append(losses_gen)
                self.losses_disc.append(losses_disc)
                self.accs_disc.append(accs_disc)
                pbar.update()

        self.x = th.cat(self.x, dim=0)
        self.y_true = th.cat(self.y_true, dim=0)
        self.y_fake = th.cat(self.y_fake, dim=0)
        self.losses_gen_ = [list(_l) for _l in list(zip(*self.losses_gen))]
        self.losses_disc_ = [list(_l) for _l in list(zip(*self.losses_disc))]
        self.accs_disc_ = [list(_l) for _l in list(zip(*self.accs_disc))]
        self.losses_gen = [sum(_l) / len(_l) for _l in self.losses_gen_]
        self.losses_disc = [sum(_l) / len(_l) for _l in self.losses_disc_]
        self.accs_disc = [sum(_l) / len(_l) for _l in self.accs_disc_]


def train_pix2pix(model_gen, model_disc, dataset, device,
                  dir_save, save_cps=True, save_events=True,
                  epochs=10, batch_size=8, 
                  gen_loss_option=0, gen_loss_weight=10,
                  lr=0.001,
                  val_pc=10.,
                  ):

    if not str(device)=='cpu':
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    DIR_TH_CPS = os.path.join(os.path.abspath(dir_save), 'CP')
    DIR_TB_EVENTS = os.path.join(os.path.abspath(dir_save), 'EVENTS')

    if save_cps:
        try:
            os.makedirs(os.path.abspath(DIR_TH_CPS))
            logging.info('Created checkpoint directory')
        except OSError:
            pass
    if save_events:
        try:
            os.makedirs(os.path.abspath(DIR_TB_EVENTS))
            logging.info('Created tensorboard summary directory')
        except OSError:
            pass

    ts_init = timestamp()
    n_batch_per_save = 10
    opter_gen = optim.Adam(model_gen.parameters(), lr=lr)  #, weight_decay=1e-8, momentum=0.9)
    opter_disc = optim.Adam(model_disc.parameters(), lr=lr)  #, weight_decay=1e-8, momentum=0.9)

    losses_gen_wgts = [gen_loss_weight, 1]

    gen_loss_func_candidates = [soft_dice_loss, l1_loss_sum, l1_loss_mean, l2_loss_sum, l2_loss_mean]
    gen_loss_func_candidates_names = ['DICE', 'L1Sum', 'L1Mean', 'L2Sum', 'L2Mean']
    loss_gen_func_name = gen_loss_func_candidates_names[gen_loss_option]
    losses_gen_funcs = [gen_loss_func_candidates[gen_loss_option], nn.BCELoss()]  #l1loss, soft_dice_loss

    loss_disc_func = nn.BCELoss()

    model_helper = ModelHelper(model_gen, model_disc, opter_gen, opter_disc)
    model_evaluator = ModelEvaluator(model_gen, model_disc, [losses_gen_funcs, losses_gen_wgts], loss_disc_func, device)

    ds = dataset  # FIXME

    n_dat = len(ds)
    n_val = int(len(ds) * val_pc / 100.)
    n_trn = n_dat - n_val

    steps_per_cycle = int(n_dat / batch_size)
    val_per_cycle = 2

    ds_true_trn, ds_true_val = random_split(ds, [n_trn, n_val])
    ds_fake_trn, ds_fake_val = random_split(ds, [n_trn, n_val])

    ds_loader = DataLoader(ds_true_trn, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    ds_loader_val = DataLoader(ds_true_val, batch_size=min(batch_size, n_trn), shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=DIR_TB_EVENTS, comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0
    val_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_trn}
        Validation size: {n_val}
        Checkpoints:     {save_cps}
        Device:          {device.type}
    ''')

    for epoch in range(epochs):

        epoch_loss = 0
        epoch_losses_gen = []
        with tqdm(total=len(ds_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='px') as pbar:
            for batch in ds_loader:

                x = batch['x']
                y_true = batch['y']

                type_ = torch.float32 
                x = x.to(device=device, dtype=type_)
                y_true = y_true.to(device=device, dtype=type_).unsqueeze(1)

                opter_gen.zero_grad()
                _, y_gen_fake, (y_disc_pred, y_disc_true) = model_helper.generator_arguments(x, y_true)

                losses_gen_args = [[y_gen_fake, y_true], [y_disc_pred, y_disc_true]]
                losses_gen = weighted_losses(losses_gen_funcs, losses_gen_args, losses_gen_wgts)
                loss_gen = th.sum(th.stack(losses_gen))
                loss_gen.backward()
                nn.utils.clip_grad_value_(model_gen.parameters(), 0.1)
                opter_gen.step()

                opter_disc.zero_grad()
                _, _, (y_disc_pred, y_disc_true) = model_helper.discriminator_arguments(x, y_true, global_step)
                loss_disc = loss_disc_func(y_disc_pred, y_disc_true)
                loss_disc.backward()
                nn.utils.clip_grad_value_(model_disc.parameters(), 0.1)
                opter_disc.step()

                # epoch_loss += loss.item()
                # print(x.shape)

                acc_disc = accuracy(y_disc_pred.detach(), y_disc_true.detach())

                writer.add_scalar('Train/D/Acc', acc_disc.item(), global_step)
                writer.add_scalar('Train/D/BCE', loss_disc.item(), global_step)
                writer.add_scalar('Train/G/{}'.format(loss_gen_func_name), losses_gen[0].item(), global_step)
                writer.add_scalar('Train/G/BCE', losses_gen[1].item(), global_step)

                epoch_losses_gen.append([losses_gen[0].item(), losses_gen[1].item()])

                pbar.set_postfix(**{'G: {}:'.format(loss_gen_func_name): losses_gen[0].item(), 'G: BCE:': losses_gen[1].item(), 'D: BCE:': loss_disc.item(), 'D: Acc:': acc_disc.item(), 'D is on:': ['True', 'Fake'][global_step % 2 == 0]})

                pbar.update()



                if global_step % int(steps_per_cycle / val_per_cycle) == 0: # and global_step > 0:

                    model_evaluator(ds_loader_val)
                    losses_gen_val = model_evaluator.losses_gen
                    losses_disc_val = model_evaluator.losses_disc
                    accs_disc_val = model_evaluator.accs_disc
                    x_val = model_evaluator.x
                    y_val_true = model_evaluator.y_true
                    y_val_fake = model_evaluator.y_fake

                    writer.add_scalar('Validation/D/BCEFake', losses_disc_val[0], global_step)
                    writer.add_scalar('Validation/D/BCETrue', losses_disc_val[1], global_step)
                    writer.add_scalar('Validation/D/AccFake', accs_disc_val[0], global_step)
                    writer.add_scalar('Validation/D/AccTrue', accs_disc_val[1], global_step)
                    writer.add_scalar('Validation/G/{}'.format(loss_gen_func_name), losses_gen_val[0], global_step)
                    writer.add_scalar('Validation/G/BCE', losses_gen_val[1], global_step)

                    logging.info('Val: D: BCE: on Fake {}'.format(losses_disc_val[0]))
                    logging.info('Val: D: BCE: on True {}'.format(losses_disc_val[1]))
                    logging.info('Val: D: Acc: on Fake {}'.format(accs_disc_val[0]))
                    logging.info('Val : D: Acc: on True {}'.format(accs_disc_val[1]))
                    logging.info('Val: G: {}: {}'.format(loss_gen_func_name, losses_gen_val[0]))
                    logging.info('Val: G: BCE: {}'.format(losses_gen_val[1]))

                    _ = writer.add_images('Input/True/X', x_val, global_step)
                    _ = writer.add_images('Output/True/Y', y_val_true, global_step)
                    _ = writer.add_images('Output/Fake/Y', y_val_fake, global_step)

                if val_step == 0:
                    losses_gen_best_val = [(0, losses_gen_val[0]), (0, losses_gen_val[1])]
                else:
                    if losses_gen_val[0] <= losses_gen_best_val[0][1]:
                        losses_gen_best_val[0] = ((epoch, global_step, val_step), losses_gen_val[0])
                        torch.save(model_gen.state_dict(),
                        path.join(DIR_TH_CPS_BEST, ts_init + f'torch_model_CP_gen_best_val_loss1.pth'))
                    if losses_gen_val[1] <= losses_gen_best_val[1][1]:
                        losses_gen_best_val[1] = ((epoch, global_step, val_step), losses_gen_val[1])
                        torch.save(model_gen.state_dict(),
                        path.join(DIR_TH_CPS_BEST, ts_init + f'torch_model_CP_gen_best_val_loss2.pth'))

                    val_step += 1

                global_step += 1

        epoch_losses_gen = mean(array(epoch_losses_gen), axis=1).flatten()
        if epoch == 0:
            losses_gen_best = [(0, epoch_losses_gen[0]), (0, epoch_losses_gen[1])
        else:
            if epoch_losses_gen[0] <= losses_gen_best[0][1]:
                losses_gen_best[0] = ((epoch, global_step), epoch_losses_gen[0])
                torch.save(model_gen.state_dict(),
                path.join(DIR_TH_CPS_BEST, ts_init + f'torch_model_CP_gen_best_train_loss1.pth'))
            if epoch_losses_gen[1] <= losses_gen_best[1][1]:
                losses_gen_best[1] = ((epoch, global_step), epoch_losses_gen[1])
                torch.save(model_gen.state_dict(),
                path.join(DIR_TH_CPS_BEST, ts_init + f'torch_model_CP_gen_best_train_loss2.pth'))

        if save_cps:
            torch.save(model_gen.state_dict(),
                       path.join(DIR_TH_CPS, ts_init + f'torch_model_CP_gen_epoch{epoch + 1}.pth'))
            torch.save(model_disc.state_dict(),
                       path.join(DIR_TH_CPS, ts_init + f'torch_model_CP_disc_epoch{epoch + 1}.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet-CGAN on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=10,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--loss-func', dest='loss_func', type=int, default=0,
                        help='Main loss function for generator: 0: DICE, 1: L1 Sum, 2: L1 Mean, 3: L2 Sum, 4: L2 Mean')
    parser.add_argument('-w', '--loss-func-weight', dest='loss_func_weight', type=float, default=10.,
                        help='Weight of main loss function for generator')
    # parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
    #                     help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    parser.add_argument('-a', '--use-augment', dest='use_augment', type=bool, default=True,
                        help='Use augmented data')

    return parser.parse_args()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    from core.models import ImageToBinary
    model_disc = ImageToBinary(n_channels_in=4)
    model_disc = model_disc.to(device=device)

    from core.models import UNet
    model_gen = UNet(n_channels=3, n_classes=1, bilinear=True)
    model_gen = model_gen.to(device=device)

    # load = None
    # load = ['/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/unet_gan/logs/20200506_01/CP/2020-5-6_13-35-12.051torch_model_CP_gen_epoch5.pth',
    #         '/Users/tamk/Documents/GitHub/cil-road-segmentation-2020/models/kmmt/unet_gan/logs/20200506_01/CP/2020-5-6_13-35-12.051torch_model_CP_disc_epoch5.pth']

    # model_gen.load_state_dict(torch.load(load[0]))
    # model_disc.load_state_dict(torch.load(load[1]))
    # logging.info(f'Model loaded from {load}')


    model_gen.to(device=device)
    model_disc.to(device=device)

    if args.use_augment:
        DIR_X = '../../../training/images_augmented'
        DIR_Y = '../../../training/groundtruth_augmented'
    else:
        DIR_X = '../../../training/images'
        DIR_Y = '../../../training/groundtruth'



    DIR_SAVES = './logs/' + timestamp()
    # print(DIR_SAVES)

    ds = PNGDataset(DIR_X, DIR_Y, func_x=None)

    # faster convolutions, but more memory
    # cudnn.benchmark = True
    train_pix2pix(model_gen, model_disc, ds, 
                  device=device, dir_save=DIR_SAVES, val_pc=args.val,
                  lr=args.lr, gen_loss_option=args.loss_func, gen_loss_weight=args.loss_func_weight, 
                  epochs=args.epochs, batch_size=args.batchsize, save_events=False, save_cps=False)

