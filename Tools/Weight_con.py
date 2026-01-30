import datetime
import os
import torch
from collections import OrderedDict

class Weight_tool():
    def __init__(self, Weight_root):
        
        
        #创建
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.Weight_root = Weight_root + os.sep + timestamp
        os.mkdir(self.Weight_root)

        self.Last_save_weight_path = None   #当新权重出来时，用来删除之前保存的权重

    # def Mkdir_c():
    #     pass
    


    def Merge_weight(self, model1, model2, list1, list2):
        
        weight_dic = OrderedDict()
 
        
        weight1 = model1.state_dict()
        weight2 = model2.state_dict()

        for k,v in weight1.items():

            if k in list1:
                
                weight_dic[k] = v
            else:
                weight_dic[k] = weight2[k]

        return weight_dic


    #这里保留还是有点问题哈，因为是保留的两个中的一部分。
    def Save_weight(self, weight, psnr, ssim, step):

        Str_save = self.Weight_root + os.sep + str(step) + "---" + str(psnr) + "---" + str(ssim)
        
        self.Last_save_weight_path = Str_save

        torch.save(weight, Str_save)


    def Delete_weight(self):
        os.remove(self.Last_save_weight_path)   