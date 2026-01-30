import torch
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from collections import OrderedDict
from basicsr.archs.mambair_arch import MambaIR

from basicsr.data.Selected_DATA import Test_Adapt_Dataset_train


import torch.nn.functional as F

from Tools.Test_tool import Testtool

from Tools.Weight_con import Weight_tool
from Tools.Log_con import logTool
from basicsr.data.data_sampler import EnlargedSampler


from Utils.Selecte_layer_tools import Select_tool
import time





def load_list(text_path):

    name_list = []
    var_list = []
    Mean_list = []

    file = open(text_path,"r")
    lines = file.readlines()

    for line in lines:
        
        line = line.replace("\n","")
        name, var , Mean = line.split("--||--")

        name_list.append(name)
        var_list.append(float(var))
        Mean_list.append(float(Mean))
        

    return name_list, var_list, Mean_list




def Get_name_list(Layer_name_root):

    Spe_path = Layer_name_root + os.sep + "Spe.text"
    Inv_path = Layer_name_root + os.sep + "Inv.text"


    Spe_name_list, Spe_var_list, Spe_mean_list = load_list(Spe_path)
    Inv_name_list, Inv_var_list, Inv_mean_list = load_list(Inv_path)


    return Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list


def Froze_layer(model_1, Spe_name_list):

    for name, para in model_1.named_parameters():

        if name in Spe_name_list:   
            para.requires_grad = False
            

            

def Update_weight(model_1, model_2, name_list, decay):

    m2_para = OrderedDict(model_2.named_parameters())
    m1_para = OrderedDict(model_1.named_parameters())

    for name, param in m1_para.items():
        if name in name_list:
            m2_para[name].mul_(decay).add_(param.data, alpha=1 - decay)



def Update_weight_by_var(model_1, model_2, name_list, Inv_var_list, Var_big):

    m2_para = OrderedDict(model_2.named_parameters())
    m1_para = OrderedDict(model_1.named_parameters())

    
    num_all_layer = len(Inv_var_list)


    if Var_big == True:         
        EMA_Base = 0.999
        base_num = 0.001
    else:
        EMA_Base = 0.99
        base_num = 0.01



    for name, param in m1_para.items():

        if name in name_list:
            index = name_list.index(name)
            # var = Inv_var_list[index] 
            ratio = (index / num_all_layer) 
            decay = EMA_Base + base_num * ratio
            m2_para[name].mul_(decay).add_(param.data, alpha=1 - decay)

    




def Train_pipline(All_step, LR_root, HR_root, Test_LR_root, Test_HR_root, Weight_path, Layer_name_root, LR_size, scale, Batchsize, Num_workers, Num_EMA_Per, decay, Val_per, Save_root, Log_root, Part_per):

    Weigt_control = Weight_tool(Save_root)
    Log_control = logTool(Log_root)
    Log_control.make_text_file()

    
    opt_model_source = OrderedDict([('upscale', 4), ('in_chans', 3), ('img_size', 64), ('patch_size', 8), ('img_range', 1.0), ('d_state', 16), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])
    model_1 = MambaIR(**opt_model_source).cuda()
    model_2 = MambaIR(**opt_model_source).cuda()



    Dataset_selected = Test_Adapt_Dataset_train(LR_root, HR_root, LR_size, scale)
    Target_sampler = EnlargedSampler(Dataset_selected, num_replicas=1, rank=0, ratio=100)
    Train_loader = torch.utils.data.DataLoader(Dataset_selected, batch_size=Batchsize, num_workers=Num_workers, sampler = Target_sampler)

    Test_tool = Testtool(Test_LR_root, Test_HR_root, scale, LR_size)


    #optim1
    optim_1 = torch.optim.Adam(
    model_1.parameters(),
    lr=1e-4,      #1e-4
    betas=(0 ** 0.8, 0.99 ** 0.8),)

    #optim2
    optim_2 = torch.optim.Adam(
    model_2.parameters(),
    lr=1e-4,    
    betas=(0 ** 0.8, 0.99 ** 0.8),)


    #Load weight
    param_key = 'params'
    load_net = torch.load(Weight_path, map_location=lambda storage, loc: storage)
    load_net = load_net[param_key]
    print(model_1.load_state_dict(load_net, strict=True))
    print(model_2.load_state_dict(load_net, strict=True))



   

    Par_ratio = 0.5
    Num_sample = 300
    Num_work = 1
    se_tool = Select_tool(LR_root, HR_root, LR_size, scale, Batchsize, Par_ratio, Num_sample, Num_work)


    # Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list = se_tool.Get_LP_FT_name(model_1)
    # Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list = se_tool.Get_DLDFT_name(model_1)


    #init_split
    Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list = se_tool.Get_init_name(model_1)


    #Froze_layer_by_name
    Froze_layer(model_1, Spe_name_list) 
    Froze_layer(model_2, Inv_name_list)


    Step = 0 

    Psnr_best = 0.
    # time_start = time.time()
    psnr_inti, ssim_init = Test_tool.Get_res(model_1)
    # time_end = time.time()
    # print("++++++++++++++++++++")
    # print(time_end - time_start)
    # print("====================")
    Psnr_best = psnr_inti
    
    Weigt_control.Save_weight(model_1.state_dict(), psnr_inti, ssim_init, 0)
    Log_control.Write_file(0,psnr_inti,ssim_init)



    for hr, lr in Train_loader:

        if Step > All_step:
            break

        model_1.train()
        model_2.train()

        lr = lr.cuda()
        hr = hr.cuda()

        sr_1 = model_1(lr)
        sr_2 = model_2(lr)

        loss_1 = F.l1_loss(sr_1, hr, reduction='mean')
        loss_2 = F.l1_loss(sr_2, hr, reduction='mean')

        model_1.zero_grad()
        model_2.zero_grad()
        loss_1.backward()      
        loss_2.backward()
        optim_1.step()
        optim_2.step()


        Step += 1







        if Step % Num_EMA_Per == 0:
            
            #DPU
            Update_weight_by_var(model_1, model_2, Inv_name_list, Inv_var_list, True)
            Update_weight_by_var(model_2, model_1, Spe_name_list, Spe_var_list, False)

            # Update_weight(model_1, model_2, Inv_name_list, decay)
            # Update_weight(model_2, model_1, Spe_name_list, decay)


        loss_all = loss_1 + loss_2
        print(float(loss_all.cpu().detach().numpy()))


        if Step % Val_per == 0:
            
            model_1.eval()
            # for hr, lr in Test_loader:
                
            psnr, ssim = Test_tool.Get_res(model_1) 

            print(Step, "-------", psnr, ssim)

            Log_control.Write_file(Step,psnr,ssim)

            if psnr > Psnr_best:

                Weigt_control.Delete_weight() 
                Save_Weight = Weigt_control.Merge_weight(model_1, model_2, Spe_name_list, Inv_name_list)
                Weigt_control.Save_weight(Save_Weight, psnr, ssim, Step)
                Psnr_best = psnr


        # Multi-split
        # if Step % Part_per == 0:
        #     Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list = se_tool.Get_init_name(model_1)
        #     Froze_layer(model_1, Spe_name_list)
        #     Froze_layer(model_2, Inv_name_list)



            
if __name__ == "__main__":


    All_step = 200000

    LR_size = 48
    scale = 4
    Batchsize = 4
    Num_workers = 4
    Num_EMA_Per = 10
    decay = 0.999

    Val_per = 500

    LR_root = ["/home/Dataset/SODA_DRealSR/Train/LR/P"]
    HR_root = ["/home/Dataset/SODA_DRealSR/Train/HR/P"]

    Test_LR_root = "/home/Dataset/SODA_DRealSR/Test/LR/P"
    Test_HR_root = "/home/Dataset/SODA_DRealSR/Test/HR/P"

    # Due to the consistent output channel characteristic inherent to SR networks, and in order to conserve computational resources, we directly adopt the weights of the Mambair network for training in the second stage.
    Weight_path = "/home/tangzz/Code/DG/Share_weight/classicSRx4.pth"
    Layer_name_root = ""

    Save_root = "Save_root"
    Log_root = "Log_root"


    Part_per = 5000
    # Part_LR = "/home/Dataset/SODA_DRealSR/Train/LR/P"
    # Part_HR = "/home/Dataset/SODA_DRealSR/Train/HR/P"


    Train_pipline(All_step, LR_root, HR_root, Test_LR_root, Test_HR_root, Weight_path, Layer_name_root, LR_size, scale, Batchsize, Num_workers, Num_EMA_Per, decay, Val_per, Save_root, Log_root, Part_per)





