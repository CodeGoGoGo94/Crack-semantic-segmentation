import torch.optim as optim
from torch import nn
import torch
import os
from my_dataset import MyDataSet_train, MyDataSet_500_train, MyDataSet_CFD_train, MyDataSet_YCD_train
from models import  FCN,SegNet,HED,Unet_5,Dilated_3_img
from losses import FocalLoss, BCEDiceLoss
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
from metrics import iou_score, dice_coef
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy
import albumentations as A
import torch.nn.functional as F
from test_model import test_m
from tqdm import tqdm 

transform_train = A.Compose([
    #A.CenterCrop(width=320, height=480),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5)
])


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()  

    font = xlwt.Font()  
    font.name = name  # 'Times New Roman'
    font.bold = bold
    font.color_index = 4
    font.height = height

    style.font = font

    return style

def create_excel_xls(path, sheet_name):
    attributes = ['Loss_epoch', 'iou_epoch', 'dice_epoch']
    workbook = xlwt.Workbook(encoding='utf-8')  
    sheet = workbook.add_sheet(sheet_name) 
    for i in range(0, len(attributes)):
        sheet.write(0, i, attributes[i], set_style('Times New Roman', 220, True))
    workbook.save(path)  


def write_excel_xls_append(path, value):
    index = len(value)  
    workbook = xlrd.open_workbook(path)  
    sheets = workbook.sheet_names()  
    worksheet = workbook.sheet_by_name(sheets[0])  
    rows_old = worksheet.nrows  
    new_workbook = copy(workbook) 
    new_worksheet = new_workbook.get_sheet(0)  
    for i in range(0, index):
        new_worksheet.write(rows_old, i, value[i],
                            set_style('Times New Roman', 220, True)) 
    new_workbook.save(path)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") 
print(f"using {device} device.")    

# path for saving model
weight_path=r'D:\New\simulation_1\params\ours.pth'

# Reading dataset
data_path_ = r'D:\New\simulation_1\Dataset (10cv)\cropping img 224\deepcrack'

for loop in tqdm(range(10)): # 了10fold cv
    
    data_path = f"{data_path_}\{loop}"
    
    save_img_path = r'D:\New\simulation_1\DCD_results(img)\train_image'
     
    
    data_loader = DataLoader(MyDataSet_500_train(data_path,transform_train),batch_size=2,shuffle=True)
    
    net=Dilated_3_img().to(device)
    
    
    #net = net.double()
    net.train() 
    """
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')
    """
    opt=optim.SGD(net.parameters(), lr = 0.0005, momentum=0.9) 
    scheduler = torch.optim.lr_scheduler.StepLR(opt,step_size=10,gamma = 0.8,last_epoch=-1,verbose=False) 
    loss_fun=BCEDiceLoss()
    
    epochs = 120
    # early stopping 
    best_score = None
    counter = 0
    patience = 20
    
    
    for epoch in tqdm(range(1,epochs + 1)):
        loss=[]
        iou = []
        dice = []
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)
            
            out_image = net(image)
            #out_image,d11,d12,d13,f1,d21,d22,d23,f2,d31,d32,d33,f3,f4 = net(image)
            #d0,d1,d2,d3,d4,u4,u3,u2,u1,c4,c3,c2,c1,out_image=net(image)
            #_out_image =torch.argmax(out_image,1,keepdim=True)
    
            train_loss = loss_fun(out_image.float(),segment_image)
            #side_loss = loss_fun(out_side1,segment_image) + loss_fun(out_side2,segment_image) + loss_fun(out_side3,segment_image) 
            #train_loss = train_loss + side_loss
            
            opt.zero_grad()
        
            train_loss.backward()
            opt.step()
            #scheduler.step()
            iou_ = iou_score(out_image,segment_image)
            dice_ = dice_coef(out_image,segment_image)
            
            loss.append(train_loss.item())
            iou.append(iou_)
            dice.append(dice_)
            """
            if i%40==0:
                print('epoch:{:.0f}- image:{:.0f}, train_loss: {:.4f}, iou: {:.4f}, dice:{:.4f}\n'.format(
                    epoch, i, train_loss.item()/2.0, iou_, dice_), end='')
                print('lr:{:}\n'.format(round(scheduler.get_last_lr()[0],6)), end='')
            """
     
            if i%40==0:
                trans = transforms.ToPILImage()
                _image=image[0]
                _segment_image=segment_image[0]
                _out_image=F.sigmoid(out_image[0])
                
                _image=trans(_image)
                _segment_image=trans(_segment_image)
                _out_image=trans(_out_image)
                
                
                def get_concat_h(im1, im2, im3):
                    dst = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
                    dst.paste(im1, (0, 0))
                    dst.paste(im2, (im1.width, 0))
                    dst.paste(im3, (im2.width + im1.width, 0))
                    return dst
                        
                _image = get_concat_h(_image,_segment_image,_out_image)
                save_img_root = save_img_path + '/' + str(epoch) + '-' +str(i) + '.JPEG'
                #_image.save(save_img_root) 
            
            #save_image(_segment_image,f'{save_path2}/{i}.png')
            #save_image(_out_image,f'{save_path3}/{i}.png')
    
            
        scheduler.step() 
        loss_m = np.mean(loss)
        iou_m = np.mean(iou)
        dice_m = np.mean(dice)
    
        if best_score is None:
            best_score = loss_m
        else:
            # Check if val_loss improves or not.
            if loss_m < best_score:
                # val_loss improves, we update the latest best_score, 
                # and save the current model
                best_score = loss_m
                #print(best_score)
                torch.save(net.state_dict(),weight_path)
            else:
                # val_loss does not improve, we increase the counter, 
                # stop training if it exceeds the amount of patience
                counter += 1
                if counter >= patience:
                    break
        
        print('loop:{:.0f}, epoch:{:.0f}, lr:{:.5f}, train_loss: {:.4f}, iou: {:.4f}, dice:{:.4f}, counter: {:.0f}\n'.format(
            loop, epoch, round(scheduler.get_last_lr()[0],6), loss_m, iou_m, dice_m, counter), end='\r')
    """
    test
    """
    loss_m,acc_m,pre_m,recall_m,dice_m,iou_m = test_m(epoch,data_path,loop)
    
    store_list = [epoch,loss_m,iou_m,dice_m,'test',loss_m,acc_m,pre_m,recall_m,dice_m,iou_m]
    
    store_list = [epoch,loss_m,iou_m,dice_m]  
