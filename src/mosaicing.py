import numpy as np
import cv2 as cv

def masks_CFA_bayer(rawshape,bayerpattern):
    """
    根据bayerpattern生成bayerraw的掩膜，只显示每个通道的值
    :param rawshape: (h,w)
    :param bayerttern: RGGB, GRBG, BGGR, GBRG
    :return:
    """
    assert len(rawshape)==2
    assert isinstance(bayerpattern,str) and len(bayerpattern)==4
    channels={channel:np.zeros(rawshape,dtype="bool") for channel in "RGB"} # 创建RGB三通道bool矩阵的字典
    for channel,(y,x) in zip(bayerpattern.upper(),[(0,0),(0,1),(1,0),(1,1)]): # zip将两个一一对应
        channels[channel][y::2,x::2]=1
        # print("channels[" + channel+ "]=\n",channels[channel],"\n" )
    return tuple(channels.values()) # 返回字典"RGB"键的值,加元组不许改动

def mosaicing_CFA_bayer(rgbimg,bayerpattern):
    assert len(rgbimg.shape)==3
    assert isinstance(bayerpattern, str) and len(bayerpattern) == 4
    R,G,B=rgbimg[:,:,0],rgbimg[:,:,1],rgbimg[:,:,2]
    Rm, Gm, Bm, = masks_CFA_bayer(R.shape, bayerpattern)
    bayerimg=R*Rm+G*Gm+B*Bm
    return bayerimg