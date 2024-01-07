from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.config import cfg, cfg_from_file
from miscc.utils import build_super_images,build_super_images2
from miscc.losses import sent_loss, words_loss, word_level_correlation, word_level_correlation_focus_RF, ContextBlock

from datasets import TextDataset
from datasets import prepare_data, prepare_labels
#for flower dataset, please use the fllowing dataset files
#from datasets_flower import TextDataset
#from datasets_flower import prepare_data
from DAMSM import RNN_ENCODER,CustomLSTM,MogLSTM, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import NetG,NetD,NetGOrigin
import torchvision.utils as vutils

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)

UPDATE_INTERVAL = 200
def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args

def gen_sample(text_encoder, netG, device, wordtoix):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """
    #captions = ['some horses in a field of green grass with a sky in the background']
    #captions = ['some horses in a field of green grass with a sunset in the background']
    captions = ['some horses in a field of green grass with a road in the background']
    #captions = ['some horses in a field of green grass with a mountain in the background']

    # captions = ['A herd of black and white cattle standing on a field']

    # captions = ['A herd of black and white cattle standing on a field',
    #  'A herd of black cattle standing on a field',
    #  'A herd of white cattle standing on a field',
    #  'A herd of brown cattle standing on a field',
    #  'A herd of black and white sheep standing on a field',
    #  'A herd of black sheep standing on a field',
    #  'A herd of white sheep standing on a field',
    #  'A herd of brown sheep standing on a field']

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)

    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = len(captions)
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    for step in range(50):

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        #######################################################
        # (2) Generate fake images
        ######################################################
        with torch.no_grad():
            # noise = torch.randn(1, 100) # using fixed noise
            # noise = noise.repeat(batch_size, 1)
            # use different noise
            noise = []
            for i in range(batch_size):
                noise.append(torch.randn(1, 100))
            noise = torch.cat(noise, 0)

            noise = noise.to(device)
            #fake_imgs, stage_masks = netG(noise, sent_emb)

            netG.lstm.init_hidden(noise)
            mask = (caps == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            fake_imgs, attention = netG(noise, sent_emb, words_embs, mask)

            #stage_mask = stage_masks[-1]
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            # fullpath = '%s_%3d.png' % (s_tmp,i)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)

            # # save the mask
            # # s_tmp = '%s/fm' % fake_img_save_dir
            # s_tmp = '%s/fm' % fake_img_save_dir
            # att_mask = attention[j]
            # att_mask = att_mask.sum(axis=0)
            #
            # att_mask = torch.sigmoid(att_mask)
            # im = att_mask.data.cpu().numpy()
            # # [0, 1] --> [0, 255]
            # # im = 1-im # only for better visualization
            # im = im * 255.0
            # im = im.astype(np.uint8)
            # #im = np.transpose(im, (1, 2, 0))
            # #im = np.squeeze(im, axis=2)
            # im = Image.fromarray(im)
            # fullpath = '%s_%d.png' % (s_tmp, i)
            # # fullpath = '%s_%d.png' % (s_tmp, idx)
            # im.save(fullpath)

def gen_sample_and_mask(text_encoder, netG, device, wordtoix, ixtoword):
    """
    generate sample according to user defined captions.

    caption should be in the form of a list, and each element of the list is a description of the image in form of string.
    caption length should be no longer than 18 words.
    example captions see below
    """

    #captions = ['A colorful yellow bird has wings with dark stripes and small eyes']
    #captions = ['A very tall black bird with a large black beak']
    #captions = ['This bird has mostly solid black plumage with slight green iridescent wing feathers and sharp white eyes']
    #captions = ['This brown bird has a red head and breast and a small pointy brown beak']
    #captions = ['A bird with a grey head back and wings but a bright yellow breast and belly']
    #captions = ['A small purple bird with black primaries and a thick bill']
    #captions = ['This is a middle size bird with a black cheek patch and a light yellow abdomen']
    #captions = ['This brown bird has a red head and breast and a small pointy brown beak']
    captions = ['A colorful yellow bird has wings with brown stripes and small eyes']

    # captions = ['A colorful blue bird has wings with dark stripes and small eyes',
    #             'A colorful green bird has wings with dark stripes and small eyes',
    #             'A colorful white bird has wings with dark stripes and small eyes',
    #             'A colorful black bird has wings with dark stripes and small eyes',
    #             'A colorful pink bird has wings with dark stripes and small eyes',
    #             'A colorful orange bird has wings with dark stripes and small eyes',
    #             'A colorful brown bird has wings with dark stripes and small eyes',
    #             'A colorful red bird has wings with dark stripes and small eyes',
    #             'A colorful yellow bird has wings with dark stripes and small eyes',
    #             'A colorful purple bird has wings with dark stripes and small eyes']

    # captions = ['A herd of black and white cattle standing on a field',
    #  'A herd of black cattle standing on a field',
    #  'A herd of white cattle standing on a field',
    #  'A herd of brown cattle standing on a field',
    #  'A herd of black and white sheep standing on a field',
    #  'A herd of black sheep standing on a field',
    #  'A herd of white sheep standing on a field',
    #  'A herd of brown sheep standing on a field']

    # captions = ['some horses in a field of green grass with a sky in the background',
    #  'some horses in a field of yellow grass with a sky in the background',
    #  'some horses in a field of green grass with a sunset in the background',
    #  'some horses in a field of yellow grass with a sunset in the background']

    # caption to idx
    # split string to word
    for c, i in enumerate(captions):
        captions[c] = i.split()

    caps = torch.zeros((len(captions), 18), dtype=torch.int64)

    for cl, line in enumerate(captions):
        for cw, word in enumerate(line):
            caps[cl][cw] = wordtoix[word.lower()]
    caps = caps.to(device)
    cap_len = []
    for i in captions:
        cap_len.append(len(i))

    caps_lens = torch.tensor(cap_len, dtype=torch.int64).to(device)

    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    netG.load_state_dict(torch.load(model_dir))
    netG.eval()

    batch_size = len(captions)
    s_tmp = model_dir[:model_dir.rfind('.pth')]
    fake_img_save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(fake_img_save_dir)

    for step in range(1):

        hidden = text_encoder.init_hidden(batch_size)
        words_embs, sent_emb = text_encoder(caps, caps_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

        #######################################################
        # (2) Generate fake images
        ######################################################
        with torch.no_grad():
            # noise = torch.randn(1, 100) # using fixed noise
            # noise = noise.repeat(batch_size, 1)
            # use different noise
            noise = []
            for i in range(batch_size):
                noise.append(torch.randn(1, 100))
            noise = torch.cat(noise, 0)

            noise = noise.to(device)
            #fake_imgs, stage_masks = netG(noise, sent_emb)

            netG.lstm.init_hidden(noise)
            mask = (caps == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            fake_imgs, attention = netG(noise, sent_emb, words_embs, mask)

            #stage_mask = stage_masks[-1]
        for j in range(batch_size):
            # save generated image
            s_tmp = '%s/img' % fake_img_save_dir
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            im = fake_imgs[j].data.cpu().numpy()
            #fake_image = fake_imgs[j].data.cpu()
            #fake_image = fake_imgs[j].detach().cpu()
            fake_image = fake_imgs.detach().cpu()
            # [-1, 1] --> [0, 255]
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            # fullpath = '%s_%3d.png' % (s_tmp,i)
            fullpath = '%s_%d.png' % (s_tmp, step)
            im.save(fullpath)

            attn_maps = attention[j]
            att_sze = attn_maps.size(2)

            lr_img = None
            cap_lengths = caps_lens.cpu().data.numpy()

            # img_set, _ = \
            #     build_super_images(fake_image, caps, ixtoword,
            #                        attn_maps, att_sze, lr_imgs=lr_img)

            img_set, _ = \
                  build_super_images2(fake_image, caps, cap_lengths, ixtoword,
                                attn_maps, att_sze, vis_size=256, topK=5)

            if img_set is not None:
                im = Image.fromarray(img_set)
                s_tmp = '%s/fm' % fake_img_save_dir
                fullpath = '%s_%d.png' % (s_tmp, step)
                im.save(fullpath)


def sampling(text_encoder, netG, dataloader,device):
    
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    # for coco wrap netG with DataParallel because it's trained on two 3090
    #    netG = nn.DataParallel(netG).cuda()
    # netG.load_state_dict(torch.load('../models/%s/netG_530.pth' % (cfg.CONFIG_NAME))) //for CUB
    # netG.load_state_dict(torch.load('../models/%s/netG_250.pth'%(cfg.CONFIG_NAME)))
    netG.load_state_dict(torch.load('../models/%s/netG_283.pth' % (cfg.CONFIG_NAME)))

    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)
    cnt = 0
    for i in range(5):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        for step, data in enumerate(dataloader, 0):
            #imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            imgs, captions, cap_lens, class_ids, keys, wrong_caps, \
            wrong_caps_len, wrong_cls_id, noise, word_labels = prepare_data(data)

            cnt += batch_size
            if step % 100 == 0:
                print('step: ', step)
            # if step > 50:
            #     break
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                noise = torch.randn(batch_size, 100)
                noise=noise.to(device)
                netG.lstm.init_hidden(noise)
                
                #fake_imgs = netG(noise,sent_emb)

                mask = (captions == 0)
                num_words = words_embs.size(2)
                if mask.size(1) > num_words:
                    mask = mask[:, :num_words]
                fake_imgs, attention = netG(noise, sent_emb, words_embs, mask)

            for j in range(batch_size):
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d.png' % (s_tmp,i)
                im.save(fullpath)



def train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD,state_epoch,batch_size,device):
    mkdir_p('../models/%s' % (cfg.CONFIG_NAME))

    gen_iterations = 0
    #####20220601 idea-3 use DAMSM Loss #####
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)
    #####20220601 idea-3 use DAMSM Loss #####

    #####20220601 idea-4 add word level discriminator #####
    contBlock = ContextBlock(256, 0.5)
    contBlock = contBlock.cuda()
    #####20220601 idea-4 add word level discriminator #####

    for epoch in range(state_epoch+1, cfg.TRAIN.MAX_EPOCH+1):
        torch.cuda.empty_cache()
        
        for step, data in enumerate(dataloader, 0):
            #torch.cuda.empty_cache()
            
            #imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            imags, captions, cap_lens, class_ids, keys, wrong_caps, \
            wrong_caps_len, wrong_cls_id, noise, word_labels = prepare_data(data)

            noise = noise.cuda()
            word_labels = word_labels.cuda()

            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

            imgs=imags[0].to(device)
            real_features = netD(imgs)
            output = netD.COND_DNET(real_features,sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.COND_DNET(real_features[:(batch_size - 1)], sent_emb[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # synthesize fake images
            
            noise = torch.randn(batch_size, 100)
            noise=noise.to(device)
            netG.lstm.init_hidden(noise)
            
            #fake = netG(noise,sent_emb)
            #####20220601 idea-2 add word attention #####
            mask = (captions == 0)
            num_words = words_embs.size(2)
            if mask.size(1) > num_words:
                mask = mask[:, :num_words]
            fake, attention = netG(noise, sent_emb, words_embs, mask)
            #####20220601 idea-2 add word attention #####

            # G does not need update with D
            fake_features = netD(fake.detach()) 

            errD_fake = netD.COND_DNET(fake_features,sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()          

            errD = errD_real + (errD_fake + errD_mismatch)/2.0

            #####20220601 idea-4 add word level discriminator #####
            # region_features_real, cnn_code_real = image_encoder(imgs)
            # result = word_level_correlation(region_features_real, words_embs,
            #                                 cap_lens, batch_size, class_ids, real_labels, word_labels)
            #
            # errD += result

            region_features_real, cnn_code_real = image_encoder(imgs)
            region_features_fake, cnn_code_fake = image_encoder(fake.detach())
            result, resultfake = word_level_correlation_focus_RF(region_features_real, words_embs,
                                            cap_lens, batch_size, class_ids, real_labels, word_labels,\
                                            contBlock, region_features_fake)

            errD += ((1.0/1000.0)*(result) + (1.0/100.0)*(resultfake)) #Batch size 12 nf 32
            Dlogs = ''
            Dlogs += 'word_level_correlation: real: %.2f fake: %.2f' % (result, resultfake)
            if gen_iterations % 100 == 0:
                print(Dlogs)
            #####20220601 idea-4 add word level discriminator #####


            optimizerD.zero_grad()
            optimizerG.zero_grad()
            errD.backward()
            optimizerD.step()

            #MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD(interpolated)
            out = netD.COND_DNET(features,sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                    inputs=(interpolated,sent_inter),
                                    grad_outputs=torch.ones(out.size()).cuda(),
                                    retain_graph=True,
                                    create_graph=True,
                                    only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0,grad1),dim=1)                        
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            optimizerG.zero_grad()
            d_loss.backward()
            optimizerD.step()
            
            # update G
            features = netD(fake)
            output = netD.COND_DNET(features,sent_emb)
            errG = - output.mean()

            #####20220601 idea-3 use DAMSM Loss #####
            gen_iterations += 1
            region_features, cnn_code = image_encoder(fake)
            w_loss0, w_loss1, _ = words_loss(region_features, words_embs,
                                             match_labels, cap_lens,
                                             class_ids, batch_size)
            w_loss = (w_loss0 + w_loss1) * \
                     cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(cnn_code, sent_emb,
                                         match_labels, class_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * \
                     cfg.TRAIN.SMOOTH.LAMBDA

            errG += (0.1/1.0)*(w_loss + s_loss)   # 0.1 /(num_head) nf 32
            logs = ''
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss)
            if gen_iterations % 100 == 0:
                print(logs)
            #####20220601 idea-3 use DAMSM Loss #####

            optimizerG.zero_grad()
            optimizerD.zero_grad()
            errG.backward()
            optimizerG.step()

            if step % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
                % (epoch, cfg.TRAIN.MAX_EPOCH, step, len(dataloader), errD.item(), errG.item()))

        vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % ('../imgs', epoch),
                        normalize=True)

        if epoch%10==0:
            torch.save(netG.state_dict(), '../models/%s/netG_%03d.pth' % (cfg.CONFIG_NAME, epoch))
            torch.save(netD.state_dict(), '../models/%s/netD_%03d.pth' % (cfg.CONFIG_NAME, epoch))       

    return count




if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ",args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                                base_size=cfg.TREE.BASE_SIZE,
                                transform=image_transform)
        ixtoword = dataset.ixtoword
        wordtoix = dataset.wordtoix
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:     
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                            base_size=cfg.TREE.BASE_SIZE,
                            transform=image_transform)
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #lstm = CustomLSTM(256, 256)
    #####20220601 idea-1 use MogLSTM #####
    lstm = MogLSTM(256, 256, 5)
    #####20220601 idea-1 use MogLSTM #####

    netG = NetG(cfg.TRAIN.NF, 100,lstm).to(device)
    #netG = NetGOrigin(cfg.TRAIN.NF, 100, lstm).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    #####20220601 idea-3 use DAMSM Loss #####
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_IMGNAME
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.cuda()

    for p in image_encoder.parameters():
        p.requires_grad = False
    print('Load image encoder from:', img_encoder_path)
    image_encoder.eval()
    #####20220601 idea-3 use DAMSM Loss #####

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    state_epoch=0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))  


    if cfg.B_VALIDATION:
        #count = sampling(text_encoder, netG, dataloader,device)  # generate images for the whole valid dataset
        #gen_sample(text_encoder, netG, device, wordtoix)  # generate images with description from user
        gen_sample_and_mask(text_encoder, netG, device, wordtoix, ixtoword)
        print('state_epoch:  %d'%(state_epoch))
    else:
        
        count = train(dataloader,netG,netD,text_encoder,optimizerG,optimizerD, state_epoch,batch_size,device)



        