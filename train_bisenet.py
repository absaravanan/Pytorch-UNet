import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from unet import fast_unet

from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from torchsummary import summary


from bisenet.utils.init_func import init_weight, group_weight
from bisenet.utils.pyt_utils import all_reduce_tensor
from bisenet.engine.lr_policy import PolyLR
from bisenet.engine.engine import Engine
from bisenet.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d

from bisenet.config import config
# from bisenet.dataloader import get_train_loader
from bisenet.network import BiSeNet

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=3e-5,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_img = '/home/ai/ai/data/coco/aadhaar_mask_augmented/train/images/'
    dir_mask = '/home/ai/ai/data/coco/aadhaar_mask_augmented/train/annotations/'
    dir_checkpoint = 'checkpoints/'

    

    ids = get_ids(dir_img)
    # print ("ids length", ids)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.9,
    #                       weight_decay=0.0005)

    # criterion = nn.BCELoss()
    min_kept = int(config.batch_size // len(
        engine.devices) * config.image_height * config.image_width // 16)
    criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7,
                                       min_kept=min_kept,
                                       use_weight=False)

    aux_criterion = SigmoidFocalLoss(ignore_label=255, gamma=2.0, alpha=0.25)



    model = BiSeNet(config.num_classes, is_training=True,
                    criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=nn.BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model.context_path,
                               nn.BatchNorm2d, base_lr)
    params_list = group_weight(params_list, model.spatial_path,
                               nn.BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.global_context,
                               nn.BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.arms,
                               nn.BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.refines,
                               nn.BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.heads,
                               nn.BatchNorm2d, base_lr * 10)
    params_list = group_weight(params_list, model.ffm,
                               nn.BatchNorm2d, base_lr * 10)

    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    # summary(model, (3, 224,224))

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net = model
        net.cuda()
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            # print ([i[0].shape for i in b])
            # print ([i[1].shape for i in b])
            optimizer.zero_grad()

            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            # cgts = np.array([i[2] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            # cgts = torch.from_numpy(cgts)

            # if gpu:
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()
            # cgts = cgts.cuda()

            # masks_pred = net(imgs)

            loss = model(imgs, true_masks.long())

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(2):
                optimizer.param_groups[i]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
                           
            loss.backward()
            optimizer.step()

            # masks_probs_flat = masks_pred.view(-1)

            # true_masks_flat = true_masks.view(-1)

            # loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=5,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')   
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1)
    # net = fast_unet(n_channels=3, n_classes=1)

    # summary(net, (3, 224,224))

    # if args.load:
    #     net.load_state_dict(torch.load(args.load))
    #     print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
