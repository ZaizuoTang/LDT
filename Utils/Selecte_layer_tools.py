import torch

from collections import OrderedDict
from basicsr.archs.mambair_arch import MambaIR

from basicsr.data.Selected_DATA import Test_Adapt_Dataset_train
import numpy as np
import torch.nn.functional as F


class Select_tool():
    def __init__(self, LR_root, HR_root, lr_size, scale, batchsize, ratio, num_sample, num_work):
        
        self.lr_root = LR_root
        self.hr_root = HR_root
        self.lr_size = lr_size
        self.scale = scale
        self.batchsize = batchsize
        self.ratio = ratio
        self.num_sa = num_sample
        self.num_work = num_work

                #搭建数据输入
        self.Dataset_selected = Test_Adapt_Dataset_train(self.lr_root, self.hr_root, self.lr_size, self.scale)
        self.Train_loader = torch.utils.data.DataLoader(self.Dataset_selected, batch_size=self.batchsize, num_workers=self.num_work)



    def Get_layer_name_by_model(self, model_s):

        Name_list = []

        for name,para in model_s.named_parameters():

            Name_list.append(name)


        return Name_list





    def Get_specific_name(self, Train_loader, model_s, Train_step, ratio):
        

        model_s.train()



        #用来装所有层的梯度
        Grad_all_info = dict()
        for name, para in model_s.named_parameters():
            Grad_all_info[name] = []
            
            para.requires_grad = True


        Count = 0

        for hr, lr in Train_loader:

            if Count >= Train_step:
                break
            Count += 1

            lr = lr.cuda()
            hr = hr.cuda()
            sr = model_s(lr)
        
            loss = F.l1_loss(sr, hr, reduction='mean')

            model_s.zero_grad()
            loss.backward()  #没有Update就不会更新网络参数，所以现在还是原始的网络参数

            for name, para in model_s.named_parameters():
                Grad_all_info[name].append(para.grad.cpu())   #他这是重复了很多次，字典里面包含着列表，列表里面存放着每个样本的梯度。

            print("当前进行到第：",Count,"个样本")


        # Mean_dic = dict()
        # Var_dic = dict()

        Name_list = []
        Mean_list = []
        Var_list = []

        
        #遍历所有网络层，将每个网络层的梯度计算平均值和方差
        for key_, value_ in Grad_all_info.items():

            tensor_list = Grad_all_info[key_]

            tensor_all = torch.stack(tensor_list,0)
            
            mean = torch.mean(tensor_all).cpu().numpy()
            var = torch.var(tensor_all).cpu().numpy()

            # Mean_dic[key_] = mean
            # Var_dic[key_] = var

            Name_list.append(key_)
            Mean_list.append(mean)
            Var_list.append(var)

        

        #获取一共有多少层网络
        Num_layer = len(Name_list)
        Num_spe = int(Num_layer * ratio)
        Num_inv = int(Num_layer - Num_spe)

        Mean_array = np.array(Mean_list)
        Var_array = np.array(Var_list)              #后面再尝试一下，除以均值的选取方法，感觉单纯依靠方差，可能度量单位没有对应上。

        #域-specific通常是导致错误预测的缘由，然后他的出现可能会带乱域不变特征，然后就需要将其进行分离。

        Sort_index = np.argsort(Var_array)  #从小到大进行排序  现在认为方差较小的网络，可能就越是偏向于域不变特征吧，然后就希望他不要受到域变化特征的影响。

        #将波动小的作为S反之作为I
        Spe_index= Sort_index[:Num_spe]
        Inv_index = Sort_index[Num_spe:]

        
        Spe_name_list = []
        Spe_var_list = []
        Spe_mean_list = []
        Inv_name_list = []
        Inv_var_list = []
        Inv_mean_list = []


        for i in range(Num_spe):
            Spe_name_list.append(Name_list[Spe_index[i]])
            Spe_var_list.append(Var_list[Spe_index[i]])
            Spe_mean_list.append(Mean_list[Spe_index[i]])


        for i in range(Num_inv):
            Inv_name_list.append(Name_list[Inv_index[i]])
            Inv_var_list.append(Var_list[Inv_index[i]])
            Inv_mean_list.append(Mean_list[Inv_index[i]])

        # Spe_name_list = Name_list[Spe_index]
        # Inv_name_list = Name_list[Inv_index]

        return Inv_name_list, Spe_name_list, Inv_var_list, Spe_var_list, Inv_mean_list, Spe_mean_list





    def Get_LP_FT_name(self,model):

        Spe_name_list = []
        Inv_name_list = []


        name_list = ["conv_first.weight","conv_first.bias","patch_embed.norm.weight","patch_embed.norm.bias"]


        for name, para in model.named_parameters():
            
            if ("layers" in name )or (name in name_list):
                Inv_name_list.append(name)
            else:
                Spe_name_list.append(name)
                

        return Spe_name_list, [], [], Inv_name_list, [], []


    def Get_DLDFT_name(self,model):

        Spe_name_list = []
        Inv_name_list = []


        name_list = ["layers"]


        for name, para in model.named_parameters():
            
            # print(name)

            if ("layers" in name ) or (name in name_list):
                Inv_name_list.append(name)
            else:
                Spe_name_list.append(name)
                

        return Spe_name_list, [], [], Inv_name_list, [], []

















        

    def Get_init_name(self, model):
        
        # #搭建网络加载网络权重
        # opt_model_source = OrderedDict([('upscale', 4), ('in_chans', 3), ('img_size', 64), ('patch_size', 8), ('img_range', 1.0), ('d_state', 16), ('depths', [6, 6, 6, 6, 6, 6]), ('embed_dim', 180), ('mlp_ratio', 2), ('upsampler', 'pixelshuffle'), ('resi_connection', '1conv')])
        # model_s = MambaIR(**opt_model_source).cuda()

        # #加载权重
        # param_key = 'params'
        # load_net = torch.load(self.weight_path, map_location=lambda storage, loc: storage)
        # load_net = load_net[param_key]
        # print(model_s.load_state_dict(load_net, strict=True))

        # #获取所有层的名字
        # Name_list = self.Get_layer_name_by_model(model_s)


        Inv_name_list, Spe_name_list, Inv_var_list, Spe_var_list, Inv_mean_list, Spe_mean_list = self.Get_specific_name(self.Train_loader, model, self.num_sa, self.ratio)


        return Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list








    def Get_current_name(self, model):
                
        Inv_name_list, Spe_name_list, Inv_var_list, Spe_var_list, Inv_mean_list, Spe_mean_list = self.Get_specific_name(self.Train_loader, model, self.num_sa, self.ratio)


        return Spe_name_list, Spe_var_list, Spe_mean_list, Inv_name_list, Inv_var_list, Inv_mean_list