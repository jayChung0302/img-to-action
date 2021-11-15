#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
import torchvision.models as model
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from PIL import Image
import torch
import torch.nn as nn

import sys
import os


from models import vid2img
from models import img2vid
from models import VidD
from models import InceptionI3d
from key_frame_selection import key_extraction as key_idx
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from utils import save_video
import videotransforms
from ucf_datasets import UCF101 as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of t/he batches')


parser.add_argument('--mode_dir', type=str, help='rgb or opt dataset directory', default='C:/UCF101/jpegs_256')
parser.add_argument('--mode', type=str, help='rgb or opt', default='rgb')
parser.add_argument('--split_path', type=str, help='', default='ucfTrainTestlist/')
parser.add_argument('--split', type=str, help='split way', default='01')
parser.add_argument('--root', type=str, help='frame count pickle directory', default='frame_count.pickle')

parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=10,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=224, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')

def main(opt):

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    ###### Definition of variables ######
    # Networks
    # A : img / B : vid
    netG_A2B = img2vid(opt.input_nc, opt.output_nc)
    netG_B2A = vid2img(opt.output_nc, opt.input_nc)
    # netD_A = model.resnet18()
    # netD_A.fc = nn.Linear(512, 1)
    netD_A = model.squeezenet1_1()
    netD_A.classifier[1] = nn.Conv2d(512,1, kernel_size=1, stride=1, padding=0)
    # netD_B = InceptionI3d(1)
    netD_B = VidD(opt.input_nc, 1)
    # TODO : perframe logits
    # per_frame_logits = nn.functional.interpolate(logits, 2, mode='linear', align_corners=True)

    if opt.cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    # netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    # input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)  # img
    # input_B = Tensor(opt.batchSize, opt.output_nc, 64, opt.size, opt.size)  # vid
    target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # # Dataset loader
    # transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
    #                transforms.RandomCrop(opt.size),
    #                transforms.RandomHorizontalFlip(),
    #                transforms.ToTensor(),
    #                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    video_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
                                           # videotransforms.RandomRotation(30),
                                           ])

    video_dataset = Dataset(opt.mode_dir, opt.split_path, opt.split, stage='train', mode=opt.mode,
                            pickle_dir=opt.root, transforms=video_transforms)
    dataloader = torch.utils.data.DataLoader(video_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4,
                                                 pin_memory=True)

    # Loss plot
    logger = Logger(opt.n_epochs, len(dataloader))
    ###################################

    ###### Training ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # Set model input
            vids, labels = batch
            imgs = torch.randn(opt.batchSize, 3, opt.size, opt.size)
            # print('video size : {}'.format(vids.size()))

            idxs = key_idx(vids)
            for n, idx in enumerate(idxs):
                imgs[n, :, :, :] = vids[n, :, idx, :, :].unsqueeze(0)
            # print('key image size : {}'.format(imgs.size()))
            # real_A = Variable(input_A.copy_(imgs))
            # real_B = Variable(input_B.copy_(vids))
            real_A = imgs.cuda()
            real_B = vids.cuda()
            # print('real_A size : {}, real_B size : {}'.format(real_A.size(), real_B.size()))
            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # GAN loss
            fake_B = netG_A2B(real_A.unsqueeze(2))
            pred_fake = netD_B(fake_B)
            # per_frame_logits_fake = netD_B(fake_B)
            # pred_fake = nn.functional.interpolate(per_frame_logits_fake, 1, mode='linear', align_corners=True)
            loss_GAN_A2B = criterion_GAN(pred_fake.squeeze(), target_real)
            # print('fake_B(vid) size: {}'.format(fake_B.size()))

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A.squeeze(2))
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)
            # print('fake_A(img) size: {}'.format(fake_A.size()))

            # React loss
            loss_identity_B = criterion_identity(fake_B, real_B) * 2 * 64
            loss_identity_A = criterion_identity(fake_A, real_A) * 2

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 5.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 5.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.squeeze(2).detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            # per_frame_logits_real = netD_B(real_B)
            # pred_real = nn.functional.interpolate(per_frame_logits_real, 1, mode='linear', align_corners=True)
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real.squeeze(), target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # Progress report (http://localhost:8097)
            logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B),
                        'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                        'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
                       images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B,
                               'recover_A':recovered_A, 'recover_B':recovered_B})
            if i % 500 == 50:
        # if epoch % 50 == 0:
                save_video(real_B.detach().cpu(), 'output/' + str(epoch) + 'ep_' + str(i) + 'realVid' + '.avi')
                save_video(fake_B.detach().cpu(), 'output/' + str(epoch) + 'ep_' + str(i) + 'fakeVid' + '.avi')
                save_video(recovered_B.detach().cpu(), 'output/' + str(epoch) + 'ep_' + str(i) + 'cycleVid' + '.avi')

                save_image(((real_A + 1) / 2).detach().cpu().squeeze(2),
                           'output/' + str(epoch) + 'ep_' + str(i) + 'realImg' + '.jpg')
                save_image(((fake_A + 1) / 2).detach().cpu().squeeze(2),
                           'output/' + str(epoch) + 'ep_' + str(i) + 'fakeImg' + '.jpg')
                save_image(((recovered_A + 1) / 2).detach().cpu().squeeze(2),
                           'output/' + str(epoch) + 'ep_' + str(i) + 'cycleImg' + '.jpg')

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')

    ###################################


if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    main(opt)
