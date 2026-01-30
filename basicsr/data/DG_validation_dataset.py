from torch.utils import data as data
import os
from basicsr.utils import FileClient, imfrombytes, img2tensor, tensor2img

from basicsr.data.transforms import augment, paired_random_crop


from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Test_dataset(data.Dataset):

    def __init__(self, LR_root, HR_root, LR_size, Scale):
        super(Test_dataset, self).__init__()

        
        self.Random_flip = True
        self.Random_rot = True


        self.Scale = Scale
        self.LR_root = LR_root
        self.HR_root = HR_root
        self.LR_size = LR_size
        self.HR_size = self.LR_size * self.Scale

        self.lr_name_list = sorted(os.listdir(self.LR_root))
        self.hr_name_list = sorted(os.listdir(self.HR_root))
        self.file_client = FileClient('disk')


    def __getitem__(self, Souce_index):
    
        lr_path = self.LR_root + os.sep + self.lr_name_list[Souce_index]
        hr_path = self.HR_root + os.sep + self.hr_name_list[Souce_index]


        lr_bytes = self.file_client.get(lr_path, 'lq')
        lr = imfrombytes(lr_bytes, float32=True)

        hr_bytes = self.file_client.get(hr_path, 'gt')
        hr = imfrombytes(hr_bytes, float32=True)


        # #随机裁剪：
        # hr, lr = paired_random_crop(hr, lr, self.HR_size, self.Scale)
        # hr, lr = augment([hr, lr], self.Random_flip, self.Random_rot)

        hr = hr[0:lr.shape[0] * self.Scale, 0:lr.shape[1] * self.Scale, :]

        hr, lr = img2tensor([hr, lr], bgr2rgb=True, float32=True)

        return hr, lr
        

    def __len__(self):
        return len(self.hr_name_list)  
