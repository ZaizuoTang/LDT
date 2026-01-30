from torch.utils import data as data
import os
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img

from basicsr.data.transforms import augment, paired_random_crop


from basicsr.utils.registry import DATASET_REGISTRY
import random


@DATASET_REGISTRY.register()
class Test_Adapt_Dataset_train(data.Dataset):

    def __init__(self, LR_root, HR_root, LR_size, Scale):
        super(Test_Adapt_Dataset_train, self).__init__()

        
        self.Random_flip = True
        self.Random_rot = True

        self.Scale = Scale
        self.LR_root_list = LR_root  #这里是一个列表
        self.HR_root_list = HR_root
        self.LR_size = LR_size
        self.HR_size = self.LR_size * self.Scale

        self.num_source = len(self.LR_root_list)

        self.lr_name_list = []
        self.hr_name_list = []

        # for i in range(self.num_source):
        #     # self.lr_name_list.append(sorted(os.listdir(self.LR_root_list[i]))) #列表嵌套
        #     # self.hr_name_list.append(sorted(os.listdir(self.HR_root_list[i])))

        #     self.lr_name_list += sorted(os.listdir(self.LR_root_list[i]))
        #     self.hr_name_list += sorted(os.listdir(self.HR_root_list[i]))


        self.lr_name_list, self.hr_name_list = self.Merge_data(self.LR_root_list, self.HR_root_list)



        self.file_client = FileClient('disk')

        # self.LR_root_1 = "/home/tangzz/Dataset/sony/Train/LR"
        # self.HR_root_1 = "/home/tangzz/Dataset/sony/Train/HR"
        # self.lr_name_list_1 = sorted(os.listdir(self.LR_root_1))
        # self.hr_name_list_1 = sorted(os.listdir(self.HR_root_1))
        # self.len2 = len(self.lr_name_list_1)


    def Get_abs_path(self,root):

        path_list = []
        file_name = sorted(os.listdir(root))
        for i, name in enumerate(file_name):

            path = root + os.sep + name
            path_list.append(path)

        return path_list





    def Merge_data(self,Lr_root_list,hr_root_list):
        
        HR_path_list = []
        LR_path_list = []

        Num_dataset = len(Lr_root_list)
        for i in range(Num_dataset):
            lr_root_c = Lr_root_list[i]
            hr_root_c = hr_root_list[i]

            lr_list_c = self.Get_abs_path(lr_root_c)
            hr_list_c = self.Get_abs_path(hr_root_c)

            HR_path_list += hr_list_c
            LR_path_list += lr_list_c

        return LR_path_list, HR_path_list
            
            







        







    def __getitem__(self, Souce_index):   #
    

        # lr_path = self.LR_root + os.sep + self.lr_name_list[Souce_index]
        # hr_path = self.HR_root + os.sep + self.hr_name_list[Souce_index]


        #随机选取从样本集合中

        # random_num = random.randint(0,self.num_source-1)

        # lr_path = self.LR_root_list[random_num] + os.sep + self.lr_name_list[random_num][Souce_index]
        # hr_path = self.HR_root_list[random_num]+ os.sep + self.hr_name_list[random_num][Souce_index]


        lr_path = self.lr_name_list[Souce_index]
        hr_path = self.hr_name_list[Souce_index]


        lr_bytes = self.file_client.get(lr_path, 'lq')
        lr = imfrombytes(lr_bytes, float32=True)

        hr_bytes = self.file_client.get(hr_path, 'gt')
        hr = imfrombytes(hr_bytes, float32=True)


        #随机裁剪：
        hr, lr = paired_random_crop(hr, lr, self.HR_size, self.Scale)


        hr, lr = augment([hr, lr], self.Random_flip, self.Random_rot)

        hr, lr = img2tensor([hr, lr], bgr2rgb=True, float32=True)

        return hr, lr
        

    def __len__(self):
        return len(self.hr_name_list)  


