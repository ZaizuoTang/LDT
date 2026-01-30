import os
import datetime


class logTool():
    def __init__(self, Save_root):

        self.Save_root = Save_root
        # os.mkdir(self.Log_root)
        self.text_file_path =  None


    
    def make_text_file(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        Save_file_path = self.Save_root  + os.sep + timestamp + ".txt"
        self.text_file_path = Save_file_path

        # self.text_file = open(Save_file_path,"w")  #直接刷新text文件，要是已经存在的话
        # self.text_file.close()

        
    def Write_file(self, Step, psnr, ssim):

        Str_write = "Step--" + str(Step) + "---" + str(psnr) + "---" + str(ssim)

        file_w = open(self.text_file_path,"a+")
        file_w.write(Str_write)
        file_w.write("\n")
        file_w.close()
