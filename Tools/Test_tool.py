import torch
import sys

sys.path.append("/home/tangzz/Code/DG/DG_Selected")
from basicsr.data.DG_validation_dataset import Test_dataset
from collections import OrderedDict
from basicsr.metrics import calculate_metric


import torch.nn.functional as F
from basicsr.utils import img2tensor, tensor2img

import numpy as np


class Testtool():
    
    def __init__(self, LR_root, HR_root, Scale, LR_size):

        self.LR_root = LR_root
        self.HR_root = HR_root
        self.Scale = Scale
        self.LR_size = LR_size
        
        #搭建数据输入
        self.Dataset_Test = Test_dataset(self.LR_root, self.HR_root, self.LR_size, self.Scale)
        self.Test_loader = torch.utils.data.DataLoader(self.Dataset_Test, batch_size=1, num_workers=0)

        self.opt_psnr = OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])
        self.opt_ssim = OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)])

        self.num_sample = len(self.Test_loader) 

    def Get_res(self, model_1):

        model_1.eval()

        psnr_all = 0.
        ssim_all = 0.
        with torch.no_grad():
            for hr, lr in self.Test_loader:

                lr = lr.cuda()
                # hr = hr.cuda()

                sr = self.SR_slice(model_1,lr,self.Scale)

                # sr = sr.cpu().numpy().squeeze()
                # hr = hr.numpy().squeeze()
                # sr = sr.transpose(1, 2, 0)
                # hr = hr.transpose(1, 2, 0)

                sr = tensor2img(sr.cpu())
                hr = tensor2img(hr)


                res_now = {'img':sr, "img2":hr}
                psnr_alone = calculate_metric(res_now, self.opt_psnr)
                ssim_alone = calculate_metric(res_now, self.opt_ssim)

                psnr_all += psnr_alone
                ssim_all += ssim_alone

        
        psnr_mean = psnr_all / self.num_sample
        ssim_mean = ssim_all / self.num_sample

        return psnr_mean, ssim_mean






        



    def SR_slice(self,model,lr,s,CLIP_tool=None):

        output_full = None
        _, C, h, w = lr.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(lr, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = s
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        mask_chops = []

    
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])

            mask_blank = np.zeros(shape=[H,W])
            mask_blank[top,left] = 1
            mask_chops.append(torch.tensor(mask_blank==1).unsqueeze(0))

        model.eval()
        with torch.no_grad():
            outputs = []
            for i_n,chop in enumerate(img_chops):
                if CLIP_tool!= None:
                    mask = mask_chops[i_n].cuda()
                    encod = CLIP_tool.Get_encoding_from_img(img,mask)
                    out = model(chop,encod)  # image processing of each partition
                else:
                    out = model(chop)
                    
                outputs.append(out)
            _img = torch.zeros(1, C, H * scale, W * scale)
            # merge
            for i in range(ral):
                for j in range(row):
                    top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                    left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                    if i == 0:
                        _top = slice(0, split_h * scale)
                    else:
                        _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                    if j == 0:
                        _left = slice(0, split_w * scale)
                    else:
                        _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                    _img[..., top, left] = outputs[i * row + j][..., _top, _left]
            output_full = _img


        model.train()

        _, _, h, w = output_full.size()
        output_full = output_full[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        return output_full





