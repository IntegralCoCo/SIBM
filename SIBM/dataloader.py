import torch
import os
from PIL import Image
import torchvision
import tqdm
def get_image_list(raw_image_path, clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(clear_image_path + image_file), image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, None, image_file])
    return image_list





class UWNetDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.is_train = is_train
        self.image_list = get_image_list(self.raw_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image), self.transform(clear_image), "_"
        return self.transform(raw_image), "_", image_name

    def __len__(self):
        return len(self.image_list)


#-================================================================================================================

def Ucolor_get_image_list(raw_image_path, RMT_image_path,clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(RMT_image_path + image_file),
                               os.path.join(clear_image_path + image_file), image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image,os.path.join(RMT_image_path + image_file) ,None, image_file])
    return image_list


class UcolorDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path,RMT_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.RMT_image_path=RMT_image_path
        self.is_train = is_train
        self.image_list = Ucolor_get_image_list(self.raw_image_path,self.RMT_image_path, self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, RMT_image,clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        RMT_image = Image.open(RMT_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image),self.transform(RMT_image), self.transform(clear_image), "_"
        return self.transform(raw_image),self.transform(RMT_image), "_", image_name

    def __len__(self):
        return len(self.image_list)
    #比起来就是在raw图的后面跟了一个RMT图

# =================================================================================================================================
def Mymodel_get_image_list(raw_image_path, RMT_image_path,canny_image_path,clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(RMT_image_path + image_file),os.path.join(canny_image_path + image_file),
                               os.path.join(clear_image_path + image_file), image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image,os.path.join(RMT_image_path + image_file) ,os.path.join(canny_image_path + image_file),
                               None, image_file])
    return image_list


class MymodelDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path,RMT_image_path,canny_image_path, clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.RMT_image_path=RMT_image_path
        self.canny_image_path = canny_image_path
        self.is_train = is_train
        self.image_list = Mymodel_get_image_list(self.raw_image_path,self.RMT_image_path, self.canny_image_path,self.clear_image_path, is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, RMT_image,canny_image,clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        RMT_image = Image.open(RMT_image)
        canny_image=Image.open(canny_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),self.transform(clear_image), "_"
        return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),"_", image_name

    def __len__(self):
        return len(self.image_list)
    #比起来就是在raw图的后面跟了一个RMT图和canny图


def canny_get_image_list(raw_image_path, RMT_image_path,canny_image_path,clear_image_path,labelcanny_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(RMT_image_path + image_file),os.path.join(canny_image_path + image_file),
                               os.path.join(clear_image_path + image_file), os.path.join(labelcanny_path + image_file),image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image,os.path.join(RMT_image_path + image_file) ,os.path.join(canny_image_path + image_file),
                               None, image_file])
    return image_list
class cannyDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path,RMT_image_path,canny_image_path, clear_image_path,labelcanny_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.RMT_image_path=RMT_image_path
        self.canny_image_path = canny_image_path
        self.labelcanny_path=labelcanny_path
        self.is_train = is_train
        self.image_list = canny_get_image_list(self.raw_image_path,self.RMT_image_path, self.canny_image_path,self.clear_image_path, self.labelcanny_path,is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, RMT_image,canny_image,clear_image,labelcanny_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        RMT_image = Image.open(RMT_image)
        canny_image=Image.open(canny_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            labelcanny_image=Image.open(labelcanny_image)
            return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),self.transform(clear_image),self.transform(labelcanny_image), "_"
        return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),"_", image_name

    def __len__(self):
        return len(self.image_list)
    #比起来就是在raw图的后面跟了一个RMT图和canny图





def color_get_image_list(raw_image_path, RMT_image_path,canny_image_path,color_path,clear_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(RMT_image_path + image_file),os.path.join(canny_image_path + image_file),os.path.join(color_path + image_file),
                               os.path.join(clear_image_path + image_file),image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image,os.path.join(RMT_image_path + image_file) ,os.path.join(canny_image_path + image_file),os.path.join(color_path + image_file),
                               None, image_file])
    return image_list
class colorDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path,RMT_image_path,canny_image_path, color_path,clear_image_path, transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.RMT_image_path=RMT_image_path
        self.canny_image_path = canny_image_path
        self.color_path=color_path
        self.is_train = is_train
        self.image_list = color_get_image_list(self.raw_image_path,self.RMT_image_path, self.canny_image_path,self.color_path,self.clear_image_path,is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, RMT_image,canny_image,color_image,clear_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        RMT_image = Image.open(RMT_image)
        canny_image=Image.open(canny_image)
        color_image=Image.open(color_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),self.transform(color_image),self.transform(clear_image), "_"
        return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),self.transform(color_image),"_", image_name

    def __len__(self):
        return len(self.image_list)
    #比起来就是在raw图的后面跟了一个RMT图和canny图


# # RMT_image=Image.open('G:/underwater/data/EUVP_out/input/264384_00020131.jpg')
# # # labal_image=Image.open('raw/label/2_img_.png')
# transform = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor()])
# # print(transform(RMT_image).shape)
# # # print(transform(labal_image).shape)
#
# trdataset=MymodelDataSet('G:/underwater/data/DUIE/raw/input/','G:/underwater/data/DUIE/raw/input_canny2/','G:/underwater/data/DUIE/raw/label/',None,transform,False)
# dataloder=torch.utils.data.DataLoader(trdataset,batch_size = 2,shuffle = True)
#
# for (img, rmt, canny, _, name) in dataloder:
#     #print(inp.shape)
#     print(rmt.shape)


def saliency_get_image_list(raw_image_path, RMT_image_path,canny_image_path,clear_image_path,labelcanny_path,saliency_image_path, is_train):
    image_list = []
    raw_image_list = [raw_image_path + i for i in os.listdir(raw_image_path)]
    if is_train:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image, os.path.join(RMT_image_path + image_file),os.path.join(canny_image_path + image_file),
                               os.path.join(clear_image_path + image_file), os.path.join(labelcanny_path + image_file),
                               os.path.join(saliency_image_path + image_file),image_file])
    else:
        for raw_image in raw_image_list:
            image_file = raw_image.split('/')[-1]
            image_list.append([raw_image,os.path.join(RMT_image_path + image_file) ,os.path.join(canny_image_path + image_file),
                               None, image_file])
    return image_list

class saliencyDataSet(torch.utils.data.Dataset):
    def __init__(self, raw_image_path,RMT_image_path,canny_image_path, clear_image_path, labelcanny_path, saliency_image_path ,transform, is_train=False):
        self.raw_image_path = raw_image_path
        self.clear_image_path = clear_image_path
        self.RMT_image_path=RMT_image_path
        self.canny_image_path = canny_image_path
        self.labelcanny_path=labelcanny_path
        self.saliency_image_path=saliency_image_path
        self.is_train = is_train
        self.image_list = saliency_get_image_list(self.raw_image_path,self.RMT_image_path, self.canny_image_path,self.clear_image_path,
                                               self.labelcanny_path,self.saliency_image_path,self.is_train)
        self.transform = transform

    def __getitem__(self, index):
        raw_image, RMT_image,canny_image,clear_image,labelcanny_image,saliency_image, image_name = self.image_list[index]
        raw_image = Image.open(raw_image)
        RMT_image = Image.open(RMT_image)
        canny_image=Image.open(canny_image)
        if self.is_train:
            clear_image = Image.open(clear_image)
            labelcanny_image=Image.open(labelcanny_image)
            saliency_image=Image.open(saliency_image)
            return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),self.transform(clear_image),self.transform(labelcanny_image),self.transform(saliency_image), "_"
        return self.transform(raw_image),self.transform(RMT_image), self.transform(canny_image),"_", image_name

    def __len__(self):
        return len(self.image_list)







# import cv2
# import os
# import numpy as np
# r_path='G:/underwater/data/LUSI_better/input_mask/'
# s_path='G:/underwater/data/LUSI_better/input_mask/'
#
# for item in os.listdir(r_path):
#     img=cv2.imread(r_path+item)
#     img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     _,binary = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
#     # binary = cv2.erode(binary, kernel)
#     print(binary.shape)
#     #binary = np.float32(binary)
#     cv2.imwrite(s_path+item,binary)

# dataset=UcolorDataSet()