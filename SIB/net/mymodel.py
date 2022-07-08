import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from utility.ptcolor import rgb2lab
import image
from utility.colorspace import *

class conv_Ucolor(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,kernelsize=3,padding=1,stride=1):
        super(conv_Ucolor, self).__init__()
        self.kernelsize=kernelsize
        self.padding=padding
        self.stride=stride
        self.conv1 =nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=self.kernelsize, padding=self.padding,stride=self.stride),
            nn.BatchNorm2d(out_ch),
            nn.Dropout(0.2),
            nn.ReLU()
        )
    def forward(self,x):
        out=self.conv1(x)
        return out


class MultiColorBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,is_last=False):
        super(MultiColorBlock, self).__init__()

        self.is_last=is_last
    # hsv部分-----------------------------------------------------------------------------------
        self.hsv_conv1 = conv_Ucolor(in_ch, out_ch)  # 1的输出要连过去作为5的输入
        self.hsv_conv2= conv_Ucolor(out_ch,out_ch)   #2
        self.hsv_conv3= conv_Ucolor(out_ch,out_ch)   #3
        self.hsv_conv4= nn.Conv2d(out_ch,out_ch,3,1,padding=1)

        self.hsv_conv5=conv_Ucolor(out_ch,out_ch) #5的输入是两个elementwsie的add，输出要连过去
        self.hsv_conv6 = conv_Ucolor(out_ch, out_ch)
        self.hsv_conv7 = conv_Ucolor(out_ch, out_ch)
        self.hsv_conv8 = nn.Conv2d(out_ch,out_ch,3,1,padding=1)
        self.hsv_pool = nn.MaxPool2d(2)

    # hsv部分-----------------------------------------------------------------------------------

    # lab部分---------------------------------------------------------------------------------------
        self.lab_conv1 = conv_Ucolor(in_ch, out_ch)  # 1的输出要连过去作为5的输入
        self.lab_conv2 = conv_Ucolor(out_ch, out_ch)  # 2
        self.lab_conv3 = conv_Ucolor(out_ch, out_ch)  # 3
        self.lab_conv4 = nn.Conv2d(out_ch, out_ch,3,1,padding=1)

        self.lab_conv5 = conv_Ucolor(out_ch, out_ch)  # 5的输入是两个elementwsie的add，输出要连过去
        self.lab_conv6 = conv_Ucolor(out_ch, out_ch)
        self.lab_conv7 = conv_Ucolor(out_ch, out_ch)
        self.lab_conv8 = nn.Conv2d(out_ch, out_ch,3,1,padding=1)
        self.lab_pool = nn.MaxPool2d(2)
    # lab部分---------------------------------------------------------------------------------------


    # rgb部分-----------------------------------------------------------------------------------------
        self.rgb_conv1 = conv_Ucolor(in_ch, out_ch)
        self.rgb_conv2 = conv_Ucolor(out_ch*3,out_ch)
        self.rgb_conv3 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv4 = nn.Conv2d(out_ch * 3, out_ch,3,1,padding=1)

        self.rgb_conv5 = conv_Ucolor(out_ch*3,out_ch)
        self.rgb_conv6 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv7 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv8 = nn.Conv2d(out_ch * 3, out_ch,3,1,padding=1)
        self.rgb_pool=nn.MaxPool2d(2)
    # rgb部分-------------------------------------------------------------------------------------




    def forward(self,x):
        """
        保证输入的x是0，1之间的
        输入是原图像，转换了三个color空间的图像（b,in_ch,h,w）
        输出是三种空间的特征拼接（b,out_ch*3,h,w）和对应的rgb，hsv，lab三个特征（b,out_ch,h,w）
        """

        Lab_x=rgb_to_lab(x)   #rgb to lab不用归一化，但是自己写的时候可以考虑一下归一化
        Hsv_x=rgb_to_hsv(x)

        #第一层
        hsv1=self.hsv_conv1(Hsv_x)
        lab1=self.lab_conv1(Lab_x)
        rgb1=self.rgb_conv1(x)
        rgb_res=rgb1
        rgb1=torch.cat([rgb1,hsv1,lab1],dim=1)

        #第二层
        hsv2=self.hsv_conv2(hsv1)
        lab2=self.lab_conv2(lab1)
        rgb2=self.rgb_conv2(rgb1)
        rgb2=torch.cat([rgb2,hsv2,lab2],dim=1)

        #第三层
        hsv3=self.hsv_conv3(hsv2)
        lab3=self.lab_conv3(lab2)
        rgb3=self.rgb_conv3(rgb2)
        rgb3=torch.cat([rgb3,hsv3,lab3],dim=1)

        #第四层
        hsv4 = self.hsv_conv4(hsv3)
        hsv4 = hsv4+hsv1
        lab4 = self.lab_conv4(lab3)
        lab4 = lab4+lab1
        rgb4 = self.rgb_conv4(rgb3)
        rgb4 = rgb4+rgb_res
        rgb4 = torch.cat([rgb4, hsv4, lab4], dim=1)

        #第五层
        hsv5 = self.hsv_conv5(hsv4)
        lab5 = self.lab_conv5(lab4)
        rgb5 = self.rgb_conv5(rgb4)
        rgb_res2=rgb5
        rgb5 = torch.cat([rgb5, hsv5, lab5], dim=1)

        #第六层
        hsv6 = self.hsv_conv6(hsv5)
        lab6 = self.lab_conv6(lab5)
        rgb6 = self.rgb_conv6(rgb5)
        rgb6 = torch.cat([rgb6, hsv6, lab6], dim=1)

        #第七层
        hsv7 = self.hsv_conv7(hsv6)
        lab7 = self.lab_conv7(lab6)
        rgb7 = self.rgb_conv7(rgb6)
        rgb7 = torch.cat([rgb7, hsv7, lab7], dim=1)

        #第八层
        hsv8 = self.hsv_conv8(hsv7)
        hsv8=hsv8+hsv5
        lab8 = self.lab_conv8(lab7)
        lab8=lab8+lab5
        rgb8 = self.rgb_conv8(rgb7)
        rgb8=rgb8+rgb_res2

        output_pre=torch.cat([rgb8,hsv8,lab8],dim=1)

        if self.is_last:
            rgb_out=rgb8
            hsv_out=hsv8
            lab_out=lab8

        else:
            rgb_out = self.rgb_pool(rgb8)
            hsv_out = self.hsv_pool(hsv8)
            lab_out = self.lab_pool(lab8)


        return rgb_out,hsv_out,lab_out,output_pre
    #写的时候就是：multi


class MultiColorBlock_inter(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch,is_last=False):
        super(MultiColorBlock_inter, self).__init__()

        self.is_last=is_last
    # hsv部分-----------------------------------------------------------------------------------
        self.hsv_conv1 = conv_Ucolor(in_ch, out_ch)  # 1的输出要连过去作为5的输入
        self.hsv_conv2= conv_Ucolor(out_ch,out_ch)   #2
        self.hsv_conv3= conv_Ucolor(out_ch,out_ch)   #3
        self.hsv_conv4= nn.Conv2d(out_ch,out_ch,3,1,padding=1)

        self.hsv_conv5=conv_Ucolor(out_ch,out_ch) #5的输入是两个elementwsie的add，输出要连过去
        self.hsv_conv6 = conv_Ucolor(out_ch, out_ch)
        self.hsv_conv7 = conv_Ucolor(out_ch, out_ch)
        self.hsv_conv8 = nn.Conv2d(out_ch,out_ch,3,1,padding=1)
        self.hsv_pool = nn.MaxPool2d(2)

    # hsv部分-----------------------------------------------------------------------------------

    # lab部分---------------------------------------------------------------------------------------
        self.lab_conv1 = conv_Ucolor(in_ch, out_ch)  # 1的输出要连过去作为5的输入
        self.lab_conv2 = conv_Ucolor(out_ch, out_ch)  # 2
        self.lab_conv3 = conv_Ucolor(out_ch, out_ch)  # 3
        self.lab_conv4 = nn.Conv2d(out_ch, out_ch,3,1,padding=1)

        self.lab_conv5 = conv_Ucolor(out_ch, out_ch)  # 5的输入是两个elementwsie的add，输出要连过去
        self.lab_conv6 = conv_Ucolor(out_ch, out_ch)
        self.lab_conv7 = conv_Ucolor(out_ch, out_ch)
        self.lab_conv8 = nn.Conv2d(out_ch, out_ch,3,1,padding=1)
        self.lab_pool = nn.MaxPool2d(2)
    # lab部分---------------------------------------------------------------------------------------


    # rgb部分-----------------------------------------------------------------------------------------
        self.rgb_conv1 = conv_Ucolor(in_ch, out_ch)
        self.rgb_conv2 = conv_Ucolor(out_ch*3,out_ch)
        self.rgb_conv3 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv4 = nn.Conv2d(out_ch * 3, out_ch,3,1,padding=1)

        self.rgb_conv5 = conv_Ucolor(out_ch*3,out_ch)
        self.rgb_conv6 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv7 = conv_Ucolor(out_ch * 3, out_ch)
        self.rgb_conv8 = nn.Conv2d(out_ch * 3, out_ch,3,1,padding=1)
        self.rgb_pool=nn.MaxPool2d(2)
    # rgb部分-------------------------------------------------------------------------------------




    def forward(self,x,Lab_x,Hsv_x):
        """
        保证输入的x是0，1之间的
        输入是原图像，转换了三个color空间的图像（b,in_ch,h,w）
        输出是三种空间的特征拼接（b,out_ch*3,h,w）和对应的rgb，hsv，lab三个特征（b,out_ch,h,w）
        """

        #第一层
        hsv1=self.hsv_conv1(Hsv_x)
        lab1=self.lab_conv1(Lab_x)
        rgb1=self.rgb_conv1(x)
        rgb_res=rgb1
        rgb1=torch.cat([rgb1,hsv1,lab1],dim=1)

        #第二层
        hsv2=self.hsv_conv2(hsv1)
        lab2=self.lab_conv2(lab1)
        rgb2=self.rgb_conv2(rgb1)
        rgb2=torch.cat([rgb2,hsv2,lab2],dim=1)

        #第三层
        hsv3=self.hsv_conv3(hsv2)
        lab3=self.lab_conv3(lab2)
        rgb3=self.rgb_conv3(rgb2)
        rgb3=torch.cat([rgb3,hsv3,lab3],dim=1)

        #第四层
        hsv4 = self.hsv_conv4(hsv3)
        hsv4 = hsv4+hsv1
        lab4 = self.lab_conv4(lab3)
        lab4 = lab4+lab1
        rgb4 = self.rgb_conv4(rgb3)
        rgb4 = rgb4+rgb_res
        rgb4 = torch.cat([rgb4, hsv4, lab4], dim=1)

        #第五层
        hsv5 = self.hsv_conv5(hsv4)
        lab5 = self.lab_conv5(lab4)
        rgb5 = self.rgb_conv5(rgb4)
        rgb_res2=rgb5
        rgb5 = torch.cat([rgb5, hsv5, lab5], dim=1)

        #第六层
        hsv6 = self.hsv_conv6(hsv5)
        lab6 = self.lab_conv6(lab5)
        rgb6 = self.rgb_conv6(rgb5)
        rgb6 = torch.cat([rgb6, hsv6, lab6], dim=1)

        #第七层
        hsv7 = self.hsv_conv7(hsv6)
        lab7 = self.lab_conv7(lab6)
        rgb7 = self.rgb_conv7(rgb6)
        rgb7 = torch.cat([rgb7, hsv7, lab7], dim=1)

        #第八层
        hsv8 = self.hsv_conv8(hsv7)
        hsv8=hsv8+hsv5
        lab8 = self.lab_conv8(lab7)
        lab8=lab8+lab5
        rgb8 = self.rgb_conv8(rgb7)
        rgb8=rgb8+rgb_res2
        output_pre = torch.cat([rgb8, hsv8, lab8], dim=1)

        if self.is_last:
            rgb_out=rgb8
            hsv_out=hsv8
            lab_out=lab8

        else:
            rgb_out = self.rgb_pool(rgb8)
            hsv_out = self.hsv_pool(hsv8)
            lab_out = self.lab_pool(lab8)


        return rgb_out,hsv_out,lab_out,output_pre


class MultiColor(nn.Module):
    """
    由开头的MulitiColorblock和后面的Multicolorblock_inter组成
    """

    def __init__(self, in_ch=3, first_ch=128):
        super(MultiColor, self).__init__()
        self.block1=MultiColorBlock(in_ch,first_ch)
        self.block2=MultiColorBlock_inter(first_ch,first_ch*2)
        self.block3=MultiColorBlock_inter(first_ch*2,first_ch*4,True)

    def forward(self,x):

        rgb128,hsv128,lab128,output128_pre=self.block1(x)
        rgb256,hsv256,lab256,output256_pre=self.block2(rgb128,hsv128,lab128)
        _,_,_,output512_pre=self.block3(rgb256,hsv256,lab256)
        return output512_pre,output256_pre,output128_pre


class channel_attention_block(nn.Module):
    """
    就是原文里的注意力通道机制的那个模块
    """
    def __init__(self, in_ch, out_ch,ratio=16):
        super(channel_attention_block, self).__init__()
        self.Globalavg=nn.AdaptiveAvgPool2d(1)
        #reshape一下
        self.dense_conect=nn.Sequential(
            nn.Linear(in_ch,int(in_ch/ratio)),
            nn.ReLU(),
            nn.Linear(int(in_ch/ratio),in_ch),
            nn.Sigmoid()
        )
        #reshape回去
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        """
        input是多个颜色空间合起来的特征
        """

        x1=self.Globalavg(x)   # (3,channel,H,W)  to (3,channel,1,1)
        x2=torch.reshape(x1,x1.shape[0:2])   # (3,channel)
        x3=self.dense_conect(x2)
        x4=torch.reshape(x3,[x3.shape[0],x3.shape[1],1,1])  #(3,channel,1,1)
        x5=x4*x
        out=self.conv(x5)

        return out


class channel_attention(nn.Module):
    """
    接受三个输入,产生三个输出
    """
    def __init__(self, in_ch=[1536,768,384], out_ch=[512,256,128]):
        super(channel_attention, self).__init__()
        self.block1=channel_attention_block(in_ch[0],out_ch[0])  # 对应output512
        self.block2=channel_attention_block(in_ch[1],out_ch[1])  # 对应output256
        self.block3=channel_attention_block(in_ch[2], out_ch[2])  # 对应output128

    def forward(self,out512,out256,out128):
        channel_out512=self.block1(out512)
        channel_out256 = self.block2(out256)
        channel_out128=self.block3(out128)
        return channel_out512,channel_out256,channel_out128


class Decoder(nn.Module):
    """
    接受三个尺寸的颜色输入和一个深度图输入和边缘图输入
    """
    def __init__(self, in_ch=[512,256,128], out_ch=3,ratio=16):
        super(Decoder, self).__init__()
        self.in_ch=in_ch
        self.pool1=nn.MaxPool2d(2)
        self.pool2=nn.MaxPool2d(2)

        self.fst_conv512=conv_Ucolor(in_ch[0],in_ch[0])
        self.fst_conv512_block=nn.Sequential(
            conv_Ucolor(in_ch[0], in_ch[0]),
            conv_Ucolor(in_ch[0], in_ch[0]),
            nn.Conv2d(in_ch[0], in_ch[0],kernel_size=3,padding=1,stride=1)
        )
        self.sec_conv512 = conv_Ucolor(in_ch[0], in_ch[0])
        self.sec_conv512_block = nn.Sequential(
            conv_Ucolor(in_ch[0], in_ch[0]),
            conv_Ucolor(in_ch[0], in_ch[0]),
            nn.Conv2d(in_ch[0], in_ch[0], kernel_size=3, padding=1, stride=1)
        )


        self.fst_conv256=conv_Ucolor(in_ch[0]+in_ch[1],in_ch[1])
        self.fst_conv256_block=nn.Sequential(
            conv_Ucolor(in_ch[1], in_ch[1]),
            conv_Ucolor(in_ch[1], in_ch[1]),
            nn.Conv2d(in_ch[1], in_ch[1],kernel_size=3,padding=1,stride=1)
        )
        self.sec_conv256 = conv_Ucolor(in_ch[1], in_ch[1])
        self.sec_conv256_block = nn.Sequential(
            conv_Ucolor(in_ch[1], in_ch[1]),
            conv_Ucolor(in_ch[1], in_ch[1]),
            nn.Conv2d(in_ch[1], in_ch[1], kernel_size=3, padding=1, stride=1)
        )



        self.fst_conv128=conv_Ucolor(in_ch[1]+in_ch[2],in_ch[2])
        self.fst_conv128_block=nn.Sequential(
            conv_Ucolor(in_ch[2], in_ch[2]),
            conv_Ucolor(in_ch[2], in_ch[2]),
            nn.Conv2d(in_ch[2], in_ch[2],kernel_size=3,padding=1,stride=1)
        )
        self.sec_conv128 = conv_Ucolor(in_ch[2], in_ch[2])
        self.sec_conv128_block = nn.Sequential(
            conv_Ucolor(in_ch[2], in_ch[2]),
            conv_Ucolor(in_ch[2], in_ch[2]),
            nn.Conv2d(in_ch[2], in_ch[2], kernel_size=3, padding=1, stride=1)
        )


        self.edge_encoder=edge_Encoder(in_ch)


        self.up1=torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

        self.last_conv=nn.Conv2d(in_ch[2],3,kernel_size=3,stride=1,padding=1)

    def forward(self,x1,x2,x3,x_d,x_e):
        """
        接受通道数为512，256，128的三个输入x1，x2，x3，和深度图x_d(batchsize,1,H,W)
        输出为最终结果（batchsize,3,H,W）
        """
        x_d=1-x_d   #1,256,256
        x512_d=x_d
        x256_d=x_d
        x128_d=x_d
        for i in range(self.in_ch[2]-1):
            x128_d=torch.cat([x128_d,x_d],dim=1)
        x256_d=torch.cat([x128_d,x128_d],dim=1)   #256,256,256
        x512_d=torch.cat([x256_d,x256_d],dim=1)   #512,256,256
        x256_d=self.pool1(x256_d)                #256,128,128
        x512_d=self.pool2(self.pool2(x512_d))    #512,64,64


        x512_e,x256_e,x128_e=self.edge_encoder(x_e)

        a1=x1*x512_d+x1                    # x1 512,64,64
        a2=self.fst_conv512(a1)
        a3=self.fst_conv512_block(a2)
        a3=a3+a3*x512_e                 # 融入高频图信息
        a4=self.sec_conv512(a2+a3)
        a5=self.sec_conv512_block(a4)
        a5=a5+a4
        a6=self.up1(a5)



        b1=x2*x256_d+x2                # x2 256,64,64
        b1=torch.cat([b1,a6],dim=1)
        b2=self.fst_conv256(b1)
        b3=self.fst_conv256_block(b2)
        b3=b3+b3*x256_e                # 融入高频图信息
        b4=self.sec_conv256(b2+b3)
        b5=self.sec_conv256_block(b4)
        b5=b5+b4
        b6=self.up2(b5)



        c1=x3*x128_d+x3
        c1=torch.cat([c1,b6],dim=1)
        c2=self.fst_conv128(c1)
        c3=self.fst_conv128_block(c2)
        c3=c3+c3*x128_e                # 融入高频图像信息
        c4=self.sec_conv128(c2+c3)
        c5=self.sec_conv128_block(c3)
        c6=c5+c4




        result=self.last_conv(c6)
        return result



class edge_Encoder(nn.Module):
    """
    input：rgb图（b,3,H,W）和深度图（b,1,H,W）
    output:最终结果（b,3,H,W）
    """

    def __init__(self, in_ch=[512, 256, 128],  ratio=16):
        super(edge_Encoder, self).__init__()
        self.conv1=nn.Sequential(
            conv_Ucolor(1,in_ch[2]),
            conv_Ucolor(in_ch[2],in_ch[2])
        )
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Sequential(
            conv_Ucolor(in_ch[2],in_ch[1]),
            conv_Ucolor(in_ch[1],in_ch[1])
        )
        self.pool2=nn.MaxPool2d(2)
        self.conv3=nn.Sequential(
            conv_Ucolor(in_ch[1],in_ch[0]),
            conv_Ucolor(in_ch[0],in_ch[0])
        )

    def forward(self,x):
        # x是canny算子产生的边缘图（batch，1，H，W）
        x1=self.conv1(x)
        result1=x1      #128,256,256
        x2=self.conv2(self.pool1(x1))
        result2=x2      #256,128,128
        x3=self.conv3(self.pool2(x2))
        result3=x3      #512,64,64

        return x3,x2,x1




class Mymodel(nn.Module):
    """
    input：rgb图（b,3,H,W）和深度图（b,1,H,W）
    output:最终结果（b,3,H,W）
    """

    def __init__(self, in_ch=[512, 256, 128], out_ch=3, ratio=16):
        super(Mymodel, self).__init__()
        self.in_ch=in_ch
        self.color_embedding=MultiColor(first_ch=in_ch[2])
        self.channel_att=channel_attention([self.in_ch[0]*3,self.in_ch[1]*3,self.in_ch[2]*3], out_ch=self.in_ch)
        self.decoder=Decoder( in_ch=self.in_ch)

    def forward(self,x,x_d,x_e):
        x512_pre,x256_pre,x128_pre=self.color_embedding(x)
        y512,y256,y128=self.channel_att(x512_pre,x256_pre,x128_pre)
        result=self.decoder(y512,y256,y128,x_d,x_e)

        return result

# device=torch.device('cuda:0')
# x=torch.randn(1,3,100,100)
# x_d=torch.randn(1,1,100,100)
# x_e=torch.randn(1,1,100,100)
# model=Mymodel()
# model.cuda()
# x=x.to(device)
# x_d=x_d.to(device)
# x_e=x_e.to(device)
# r=model(x,x_d,x_e)
# print(r.shape)