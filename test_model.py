import os
import time
import torch
from models import FCN,SegNet,HED,Unet_5,Dilated_3_img
from my_dataset import MyDataSet_test, MyDataSet_500_test, MyDataSet_CFD_test, MyDataSet_YCD_test
from torch.utils.data import DataLoader
from losses import FocalLoss, BCEDiceLoss
from metrics import iou_score, dice_coef, BinaryMetrics
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import xlwt
import xlrd
from xlutils.copy import copy
import albumentations as A
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image


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

def test_m(epoch,data_path,loop):
    epoch = epoch
    data_path = data_path
    loop = loop
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu") 
    
   
    save_img_path = r'D:\New\simulation_1\params\out_image\our_good_2'
    
    data_loader = DataLoader(MyDataSet_test(data_path),batch_size=1,shuffle=False)
    net=Dilated_3_img().to(device)
    
    #net = net.double() 
    weights=r'D:\New\simulation_1\params\ours.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')
        
           
    
    loss_fun = BCEDiceLoss()
    
    loss=[]
    iou = []
    dice = []
    pre = []
    recall = []
    f1 = []
    acc = []
    
    net.eval()
    time_spend = []
    with torch.no_grad(): 
        #for count in range(21): 
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)
            
            #print(image.size())
            #print(segment_image.size())
            
            start = time.time()
            out_image=net(image)
            #d0,d1,d2,d3,d4,u4,u3,u2,u1,c4,c3,c2,c1,out_image=net(image)
            #out_image,d11,d12,d13,f1,d21,d22,d23,f2,d31,d32,d33,f3,f4=net(image)
            
            end = time.time()
            duration = (end)-(start)
            time_spend.append(duration)
            
            
            test_loss=loss_fun(out_image,segment_image)
            iou_ = iou_score(out_image,segment_image)
            dice_ = dice_coef(out_image,segment_image)
            
            acc_, _, pre_, _, recall_ = BinaryMetrics()._calculate_overlap_metrics(gt=segment_image, pred=out_image)
            #pre.append(pre_)
            #recall_.append(recall_)
            acc.append(float(acc_.cpu().numpy()))
            loss.append(test_loss.item())
            iou.append(iou_)
            dice.append(dice_)
            pre.append(float(pre_.cpu().numpy()))
            recall.append(float(recall_.cpu().numpy()))
            """
            if i%10==0:
                print('image: {:}, test_loss: {:.3f}, acc: {:.3f}, pre: {:.3f}, recall: {:.3f},dice:{:.3f}, iou: {:.3f},\n'.format(i,
                    test_loss.item(),acc_, pre_, recall_, dice_, iou_), end='')
            """
   
            
            def binarymap(img):
                _img=img[0]
                _img = F.sigmoid(_img)
                #_img = torch.where(_img<0.5, torch.zeros_like(_img),torch.ones_like(_img))
                trans = transforms.ToPILImage()
                _img=trans(_img)
                return _img
            
            def binarymap1(img, triger=False):
                _img=img[0]
                if triger:
                    _img = F.sigmoid(_img)
               # _img = torch.where(_img<0.5, torch.zeros_like(_img),torch.ones_like(_img))
                trans = transforms.ToPILImage()
                _img=trans(_img)
                return _img
            
            _image = binarymap1(image)
            _segment_image = binarymap1(segment_image)
            
            _out_image = binarymap1(out_image,True)
            
            '''
            _d11 = binarymap(d11) # 
            _d12 = binarymap(d12)
            _d13 = binarymap(d13)
            _f1 = binarymap(f1)
            _d21 = binarymap(d21)
            _d22 = binarymap(d22)
            _d23 = binarymap(d23)
            _f2 = binarymap(f2)
            _d31 = binarymap(d31)
            _d32 = binarymap(d32)
            _d33 = binarymap(d33)
            _f3 = binarymap(f3)
            _f4 = binarymap(f4)
            
            
            _d0 = binarymap(d0) # unet
            _d1 = binarymap(d1)
            _d2 = binarymap(d2)
            _d3 = binarymap(d3)
            _d4 = binarymap(d4)
            _u4 = binarymap(u4)
            _u3 = binarymap(u3)
            _u2 = binarymap(u2)
            _u1 = binarymap(u1)
            _c4 = binarymap(c4)
            _c3 = binarymap(c3)
            _c2 = binarymap(c2)
            _c1 = binarymap(c1)
            '''
            
            '''
            trans = transforms.ToPILImage()
            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=torch.sigmoid(out_image[0])
            _d0=(F.sigmoid(d0[0]))
            _d1=trans(F.sigmoid(d1[0]))
            _d2=trans(F.sigmoid(d2[0]))
            _d3=trans(F.sigmoid(d3[0]))
            _d4=trans(F.sigmoid(d4[0]))
            _u1=trans(F.sigmoid(u1[0]))
            _u2=trans(F.sigmoid(u2[0]))
            _u3=trans(F.sigmoid(u3[0]))
            _u4=trans(F.sigmoid(u4[0]))
            _c1=trans(torch.sigmoid(c1[0]))
            _c2=trans(F.sigmoid(c2[0]))
            _c3=trans(F.sigmoid(c3[0]))
            _c4=trans(F.sigmoid(c4[0]))
            
            #_c1 = torch.where(_d0>0.5, torch.zeros_like(_c1),torch.ones_like(_c1))
            
            #_c1=trans(_c1)
            
            _image=trans(_image)
            _segment_image=trans(_segment_image)
            _out_image=trans(_out_image)
            '''
            
            def get_concat_h(im1, im2, im3, im4, im5, im6, im7, im8, im9, im10, im11, im12, im13, im14, im15, im16):
                dst = Image.new('RGB', (im1.width*4 , im1.height*4))
                dst.paste(im1, (0, 0))
                dst.paste(im2, (im1.width, 0))
                dst.paste(im3, (im1.width*2, 0))
                dst.paste(im4, (im1.width*3, 0))
                dst.paste(im5, (0, im1.width))
                dst.paste(im6, (im1.width, im1.width))
                dst.paste(im7, (im1.width*2, im1.width))
                dst.paste(im8, (im1.width*3, im1.width))
                dst.paste(im9, (0, im1.width*2))
                dst.paste(im10, (im1.width, im1.width*2))
                dst.paste(im11, (im1.width*2, im1.width*2))
                dst.paste(im12, (im1.width*3, im1.width*2))
                dst.paste(im13, (0, im1.width*3))
                dst.paste(im14, (im1.width, im1.width*3))
                dst.paste(im15, (im1.width*2, im1.width*3))
                dst.paste(im16, (im1.width*3, im1.width*3))
                return dst
                   
            #_image = get_concat_h(_image,_segment_image,_out_image,_f4,_d11,_d12,_d13,_f1,_d21,_d22,_d23,_f2,_d31,_d32,_d33,_f3)
            #_image = get_concat_h(_image,_segment_image,_out_image,_d0,_d1,_d2,_d3,_d4,_u4,_u3,_u2,_u1,_c4,_c3,_c2,_c1)

            #_image = _c1
            #save_img_root = save_img_path + '/' +'Unet (thin)' + '-' +str(epoch)+ '-' +str(i) + '-' + str(round(dice_,2)) + '-'+ str(round(iou_,2)) + '.JPEG'
            #_image.save(save_img_root)
            
        acc_m = np.mean(acc)    
        loss_m = np.mean(loss)
        iou_m = np.mean(iou)
        dice_m = np.mean(dice)
        recall_m = np.mean(recall)
        pre_m = np.mean(pre)
        time_spend = time_spend+[0.00001]
        write_excel_xls_append(r"D:\New\simulation_1\params\10cv_results.xls", [loop,epoch,loss_m,acc_m,pre_m,recall_m,dice_m,iou_m,len(time_spend[:])/sum(time_spend[:])])                                                                        
        #print(count)
            #print('fps:{:.2f}\n'.format(len(time_spend[1:-1])/sum(time_spend[1:-1])))
    return loss_m,acc_m,pre_m,recall_m,dice_m,iou_m
