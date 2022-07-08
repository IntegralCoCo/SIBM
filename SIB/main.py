import numpy as np
import torch
from torch.nn import Module
import torchvision
from torchvision import transforms

import wandb
import os
os.environ["WANDB_API_KEY"] = " "  #your key
import argparse
from dataclasses import dataclass
from tqdm.autonotebook import tqdm, trange
from dataloader import MymodelDataSet,saliencyDataSet
from metrics_calculation import *
from model import *
from combined_loss import *
import datetime
import sys
import SIB_Net
__all__ = [
    "Trainer",
    "setup",
    "training",
]

## TODO: Update config parameter names
## TODO: remove wandb steps
## TODO: Add comments to functions

@dataclass
class Trainer:
    model: Module
    opt: torch.optim.Optimizer
    loss: Module

    @torch.enable_grad()
    def train(self, train_dataloader, config, test_dataloader = None):
        device = config['device']
        print("config['device']为"+str(config['device']))
        primary_loss_lst = []
        vgg_loss_lst = []
        total_loss_lst = []
        lab_loss_lst = []
        lch_loss_lst = []
        ssim_loss_lst = []
        # 加载已经训练的模型
        if config.use_pretrain:
            # Load pretrained models
            self.model.load_state_dict(torch.load(config.checkpoint))
            # discriminator.load_state_dict(torch.load("saved_models/D/discriminator_%d.pth" % (start_epoch)))
            print('successfully loading moedl {} 成功！'.format(config.checkpoint))
        else:
            print('No pretrain model found, training will start from scratch！')

        UIQM, UICM, UISM, UICONM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
        wandb.log({f"[Test] Epoch": 0,
                   "[Test] UIQM": np.mean(UIQM),
                   "[Test] SSIM": np.mean(SSIM),
                   "[Test] PSNR": np.mean(PSNR),
                   "[Test] UICM": np.mean(UICM),
                   "[Test] UISM": np.mean(UISM),
                   "[Test] UICONM": np.mean(UICONM), },
                  commit=True
                  )
        # log初始化
        log_path_suffix = 'test_logs/' + 'v20_UIEB'
        file_path = log_path_suffix + '.log'
        f = open(file_path, 'a')
        sys.stdout = open(file_path, 'a', encoding='utf-8')
        sys.stdout.write('-------------------------------------------------------------------------------')
        self.opt.zero_grad()
        #scheduler_opt = torch.optim.lr_scheduler.StepLR(self.opt, step_size=50, gamma=0.9)
        last_psnr = 0.420001
        last_SSIM = 0.0400001
        e_loss=nn.MSELoss()
        for epoch in trange(0,config.num_epochs,desc = f"[Full Loop]", leave = False):
            primary_loss_tmp = 0
            vgg_loss_tmp = 0             #vgg loss的值是乘了系数的
            total_loss_tmp = 0
            ssim_loss_tmp = 0
            lab_loss_tmp = 0
            lch_loss_tmp = 0

            ire=0
            for inp, rmt ,canny, label,canny_label,saliency, _ in tqdm(train_dataloader, desc = f"[Train]", leave = False):
                inp = inp.to(device)
                #rmt = rmt.to(device)
                label = label.to(device)
                canny = canny.to(device)
                canny_label=canny_label.to(device)
                saliency=saliency.to(device)
                #inp = torch.cat([inp,rmt],dim=1)

                self.model.train()


                out1,out2,pixel_out,out,lam1,lam2,lam3 = self.model(inp,canny,saliency)
                # print('成功通过model')
                loss, mse_loss, vgg_loss,ssim_loss,lab_loss,lch_loss = self.loss(out, label)
                loss_p, mse_loss_p, vgg_loss_p, ssim_loss_p, lab_loss_p, lch_loss_p = self.loss(lam1*pixel_out, lam1*label)
                loss_s, _, _, _, _, _ = self.loss(lam2 * out1,lam2 * label)
                loss_e, _, _, _, _, _  = self.loss(lam3 * out2, lam3 * label)
                loss=loss+loss_e+loss_p+loss_s
                loss=loss/config.grad_acc
                loss.backward()
                if((ire+1)%config.grad_acc==0):
                    self.opt.step()
                    #warmup_scheduler.dampen()                             # warmup改动
                    self.opt.zero_grad()
                primary_loss_tmp += mse_loss.item()
                vgg_loss_tmp += vgg_loss.item()
                ssim_loss_tmp += ssim_loss.item()
                lab_loss_tmp += lab_loss.item()
                lch_loss_tmp += lch_loss.item()
                total_loss_tmp += loss.item()
                ire=(ire+1)%len(train_dataloader)
            total_loss_lst.append(total_loss_tmp/len(train_dataloader))
            vgg_loss_lst.append(vgg_loss_tmp/len(train_dataloader))
            primary_loss_lst.append(primary_loss_tmp/len(train_dataloader))
            ssim_loss_lst.append(ssim_loss_tmp/len(train_dataloader))
            lab_loss_lst.append(lab_loss_tmp / len(train_dataloader))
            lch_loss_lst.append(lch_loss_tmp / len(train_dataloader))
            wandb.log({f"[Train] Total Loss" : total_loss_lst[epoch],
                       "[Train] Primary Loss" : primary_loss_lst[epoch],
                       "[Train] VGG Loss" : vgg_loss_lst[epoch],
                       "[Train] ssim Loss": ssim_loss_lst[epoch],
                       "[Train] lab Loss": lab_loss_lst[epoch],
                       "[Train] lch Loss": lch_loss_lst[epoch],},
                      commit = True
                      )

            if (config.test == True):
                UIQM,UICM,UISM,UICONM, SSIM, PSNR = self.eval(config, test_dataloader, self.model)
                wandb.log({f"[Test] Epoch": epoch+1,
                           "[Test] UIQM" : np.mean(UIQM),
                           "[Test] SSIM" : np.mean(SSIM),
                           "[Test] PSNR" : np.mean(PSNR),
                           "[Test] UICM" : np.mean(UICM),
                           "[Test] UISM" : np.mean(UISM),
                           "[Test] UICONM" : np.mean(UICONM),},
                          commit = True
                          )
            sys.stdout.write(
                "\r[Epoch %d/%d] , [SSIM %f] , [PSNR: %f] , [UIQM: %f], [UICM: %f] , [UISM: %f], [UICONM: %f]  "
                % (
                    epoch,
                    config['num_epochs'],
                    float(np.mean(SSIM)),
                    float(np.mean(PSNR)),
                    float(np.mean(UIQM)),
                    float(np.mean(UICM)),
                    float(np.mean(UISM)),
                    float(np.mean(UICONM)),
                )
            )
            # scheduler_opt.step()

            if epoch % config.print_freq == 0:
                print('epoch:[{}]/[{}], image loss:{}, MSE / L1 loss:{}, VGG loss:{}'.format(epoch,config.num_epochs,str(total_loss_lst[epoch]),str(primary_loss_lst[epoch]),str(vgg_loss_lst[epoch])))
                # wandb.log()

            if not os.path.exists(config.snapshots_folder):
                os.mkdir(config.snapshots_folder)

            if(np.mean(PSNR)>last_psnr)and(np.mean(SSIM)>last_SSIM):
                torch.save(self.model,config.snapshots_folder + 'v20_UIEB_best.pth')
                last_psnr = np.mean(PSNR)
                last_SSIM = np.mean(SSIM)
            if (epoch ==20):
                torch.save(self.model, config.snapshots_folder + 'v20_UIEB_epoch{}_psnr{}_ssim{}.pth'.format(epoch,np.mean(PSNR),np.mean((SSIM))))
            if (epoch % config.snapshot_freq == 0)and(epoch!=0):
                torch.save(self.model, config.snapshots_folder + 'v20_UIEB_epoch{}_psnr{}_ssim{}.pth'.format(epoch,np.mean(PSNR),np.mean((SSIM))))


    @torch.no_grad()
    def eval(self, config, test_dataloader, test_model):
        test_model.eval()
        for i, (img,sal,canny,_, name) in enumerate(test_dataloader):
            with torch.no_grad():
                img = img.to(config.device)
                sal = sal.to(config.device)
                canny=canny.to(config.device)
                generate_img,e_out = test_model(img,canny,sal)
                torchvision.utils.save_image(generate_img, config.output_images_path + name[0])
                torchvision.utils.save_image(e_out, config.outputgrad_images_path + name[0])
        SSIM_measures, PSNR_measures = calculate_metrics_ssim_psnr(config.output_images_path,config.GTr_test_images_path)
        UIQM_measures,UICM_measures,UISM_measures,UICONM_measures = calculate_UIQM(config.output_images_path)
        return UIQM_measures, UICM_measures,UISM_measures,UICONM_measures,SSIM_measures, PSNR_measures

def setup(config):
    if torch.cuda.is_available():
        config.device = "cuda"
        print('GPU加载成功！')
    else:
        config.device = "cpu"

    model =SIB_Net.Mymodel_compress_v4().to(config['device'])
    print('model的device为：')
    print(next(model.parameters()).device)
    # if(len(device_ids)>1):
    #     model=nn.DataParallel(model)

    transform = transforms.Compose([transforms.Resize((config.resize,config.resize)),transforms.ToTensor()])
    #transform = transforms.Compose([transforms, transforms.ToTensor()])
    train_dataset = saliencyDataSet(config.input_images_path,config.trainRMT_images_path,config.traincanny_images_path
    ,config.label_images_path,config.labelcanny_images_path,config.saliency_images_path,transform, True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config.train_batch_size,shuffle = True)
    print("Train Dataset Reading Completed.")

    loss = Multicombinedloss(config)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(model, opt, loss)

    if config.test:
        test_dataset = MymodelDataSet(config.test_images_path,config.testRMT_images_path ,config.testcanny_images_path,None, transform, False)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)
        print("Test Dataset Reading Completed.")
        return train_dataloader, test_dataloader, model, trainer
    return train_dataloader, None, model, trainer

def training(config):
    # Logging using wandb
    wandb.init(project = "your project name")
    wandb.config.update(config, allow_val_change = True)
    config = wandb.config


    ds_train, ds_test, model, trainer = setup(config)
    trainer.train(ds_train, config,ds_test)
    print("==================")
    print("Training complete!")
    print("==================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_images_path', type=str, default=" ",help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="",help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--test_images_path', type=str, default=" ",help='path of input images(underwater images) for testing default:./data/test/')
    parser.add_argument('--GTr_test_images_path', type=str, default=" ",help='path of input ground truth images(underwater images) for testing default:./data/test_label/')

    parser.add_argument('--trainRMT_images_path', type=str, default="/home/mp/raw_nothing/input_mask/")
    parser.add_argument('--testRMT_images_path', type=str, default="/home/mp/raw_nothing/input_mask/")

    parser.add_argument('--traincanny_images_path', type=str, default="/home/mp/raw_nothing/input_canny/", help='grad')
    parser.add_argument('--testcanny_images_path', type=str, default="/home/mp/raw_nothing/test_canny/", help='grad_label')
    parser.add_argument('--labelcanny_images_path', type=str, default="/home/mp/raw_nothing/input_canny_label/")

    parser.add_argument('--saliency_images_path', type=str, default="/home/mp/raw_nothing/input_mask/")
    parser.add_argument('--test', default=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--step_size',type=int,default=400,help="Period of learning rate decay") #50
    parser.add_argument('--num_epochs', type=int, default=1001)
    parser.add_argument('--train_batch_size', type=int, default=4,help="default : 1")
    parser.add_argument('--test_batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--gpu0_bts',type=int,default=6,help="主gpu的显存个数")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    parser.add_argument('--cuda_id', type=str, default="0",help="id of cuda device,default:0")
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--snapshot_freq', type=int, default=400)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--output_images_path', type=str, default="/home/mp/raw_nothing/output")
    parser.add_argument('--outputgrad_images_path', type=str, default="/home/mp/raw_nothing/outputgrad/")
    parser.add_argument('--use_pretrain', type=bool, default=False)
    parser.add_argument('--checkpoint', type=str, default="snapshots/。。。。")
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--vgg_para', type=float,default=0.01)
    parser.add_argument('--MSE_para', type=float, default=1)
    parser.add_argument('--ssim_para', type=float, default=1000)
    parser.add_argument('--lab_para', type=float, default=0.00001)
    parser.add_argument('--lch_para', type=float, default=0.00001)
    parser.add_argument('--L1_para', type=float, default=1)
    parser.add_argument('--grad_acc', type=int, default=4)
    parser.add_argument('--grad_para', type=float, default=1)

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_id
    # device_ids = range(torch.cuda.device_count())
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.output_images_path):
        os.mkdir(config.output_images_path)
    if not os.path.exists(config.outputgrad_images_path):
        os.mkdir(config.outputgrad_images_path)
    training(config)
