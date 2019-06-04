# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:54:35 2019

@author: zun
"""

import numpy as np
import cv2
import math
def videoProcess(frame):
#传入视频帧，返回去雾处理后的帧，即可实现实时去雾
    frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    defog=fastdehaze(frame/255.0)*255
    m=defog.astype('uint8')
    res=adjustContractAndBrithtness(m)
    return res
def fastdehaze(img):
    #没有这一步 图像就不正常
    h=img.shape[0]
    w=img.shape[1]
    M=np.zeros((h,w))
    M=img#提取rgb三通道最小值
    m_av=np.mean(M)#提取M中所有元素均值
    M_ave=cv2.blur(M,(7,7))#均值滤波
    T=np.zeros((h,w))
    e=3.0
    T= M_ave*min(0.96,e*m_av)#e是一个用来调节的参数，当e值越大时，去雾后的图像就越暗，去雾效果就越明显，e值越小时，图像偏白，有明显的雾气
    L=np.zeros((h,w))
    L=np.minimum(M,T)   
    A=0.5*(np.max(M)+np.max(M_ave))
    F=np.zeros((h,w))
    F= (img -L)/(1-L/A)  
    return F

def adjustContractAndBrithtness(img):
    contract=0.5#对比度
    brightness=0.4#亮度
    k = math.tan( (45 + 44 * contract) / 180 * math.pi )
    res = np.uint8(np.clip(((img- 127.5 * (1-brightness)) * k + 127.5 * (1+brightness)), 0, 255))##降低亮度，提升对比度
    return res

    

    
        
   