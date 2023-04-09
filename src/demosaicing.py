"""
demosaic算法有：最近邻差值；双线性差值bilinear；色差法；色比法；
一阶微分梯度法；二阶微分梯度法；hamilton自适应差值法；menon法；malvar2004法
测试方法：korda图mosaic后经过demosaic之后测psnr，ssim，保存图像
"""
import numpy as np
import scipy.ndimage as ndimage
import src.mosaicing as mosaic

def mono_demosaic_bilinear(bayerimg, bits, bayerpattern):
    bayerimg_ = bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    G_filter=np.array([[0,1,0],
                        [1,4,1],
                        [0,1,0]])/4
    RB_filter=np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])/4
    R=ndimage.convolve(padded_img*Rm,RB_filter)
    G=ndimage.convolve(padded_img*Gm,G_filter)
    B=ndimage.convolve(padded_img*Bm,RB_filter)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_color_ratio(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    G_filter=np.array([[0,1,0],
                        [1,4,1],
                        [0,1,0]])/4
    RB_filter=np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])/4
    G=ndimage.convolve(padded_img*Gm,G_filter)
    G1=np.where(G==0,1,G)
    R=G*ndimage.convolve(padded_img*Rm/G1,RB_filter)
    B=G*ndimage.convolve(padded_img*Bm/G1,RB_filter)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_color_diff(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    G_filter=np.array([[0,1,0],
                        [1,4,1],
                        [0,1,0]])/4
    RB_filter=np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])/4
    G=ndimage.convolve(padded_img*Gm,G_filter)
    R =G+ndimage.convolve((padded_img-G) * Rm, RB_filter)
    B=G+ndimage.convolve((padded_img-G) * Bm, RB_filter)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_first_gradient(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    Gh_filter=np.array([[0,0,0],
                        [1,2,1],
                        [0,0,0]])/2
    Gv_filter=np.transpose(Gh_filter)
    Gm_filter=(Gh_filter+Gv_filter)/2
    RB_filter=np.array([[1,2,1],
                        [2,4,2],
                        [1,2,1]])/4
    gradient_h=np.array([[0,0,0],
                         [1,0,-1],
                         [0,0,0]])
    gradient_v=np.transpose(gradient_h)
    #计算梯度
    delta_h=np.abs(ndimage.convolve(padded_img*Gm,gradient_h))
    delta_v=np.abs(ndimage.convolve(padded_img*Gm,gradient_v))
    delta_t=(delta_v+delta_h)/2
    #求取R/B位置的G，选择梯度较大的方向作为G值
    G=np.where(delta_h<delta_t,ndimage.convolve(padded_img*Gm,Gh_filter),padded_img*Gm)
    G=np.where(delta_v<delta_t,ndimage.convolve(padded_img*Gm,Gv_filter),G)
    G=np.where(delta_h==delta_t,ndimage.convolve(padded_img*Gm,Gm_filter),G)
    #R/B的计算为色差法
    R=G+ndimage.convolve((padded_img-G)*Rm,RB_filter)
    B=G+ndimage.convolve((padded_img-G)*Bm,RB_filter)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_second_gradient(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    h3=np.array([[0,0,0],
                [1,0,1],
                [0,0,0]])/2
    h5=np.array([[0,0,0,0,0],
                 [0,0,0,0,0],
                 [-1,0,2,0,-1],
                 [0,0,0,0,0],
                 [0,0,0,0,0]])
    v3=np.transpose(h3)
    v5=np.transpose(h5)
    RB_filter = np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]]) / 4
    #计算R/B通道的梯度
    gradient_h=np.where(np.logical_or(Rm==1,Bm==1),np.abs(ndimage.convolve(padded_img,h5)),0)
    gradient_v=np.where(np.logical_or(Rm==1,Bm==1),np.abs(ndimage.convolve(padded_img,v5)),0)
    #计算水平竖直方向G的插值
    Gh=ndimage.convolve(padded_img*Gm,h3)
    Gv=ndimage.convolve(padded_img*Gm,v3)
    #计算R/B通道的G
    G=np.where(np.logical_and((gradient_h<gradient_v),np.logical_or(Rm==1,Bm==1)),Gh,padded_img*Gm)
    G=np.where(np.logical_and((gradient_v<gradient_h),np.logical_or(Rm==1,Bm==1)),Gv,G)
    G=np.where(np.logical_and((gradient_h==gradient_v),np.logical_or(Rm==1,Bm==1)),(Gh+Gv)/2,G)
    #计算R/B,仍为色差法
    R=G+ndimage.convolve((padded_img-G)*Rm,RB_filter)
    B=G+ndimage.convolve((padded_img-G)*Bm,RB_filter)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_hamilton(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    R=padded_img*Rm
    G=padded_img*Gm
    B=padded_img*Bm
    # 计算R/B通道的G
    grad_h3=np.array([[0,0,0],
                      [1,0,-1],
                      [0,0,0]])
    grad_h5=np.array([[0,0,0,0,0],
                      [0,0,0,0,0],
                      [-1,0,2,0,-1],
                      [0,0,0,0,0],
                      [0,0,0,0,0]])
    grad_v3=np.transpose(grad_h3)
    grad_v5=np.transpose(grad_h5)
    filter_h3=np.abs(grad_h3)/2
    filter_v3=np.abs(grad_v3)/2
    filter_h5=grad_h5/4
    filter_v5=grad_v5/4
    # grad_h23=abs(G22-G24)+abs(2R23-R21-R25) grad_v23=abs(G13-G33)+abs(2R23-R03-R43)
    grad_h=np.where(np.logical_or(Rm==1,Bm==1),np.abs(ndimage.convolve(padded_img,grad_h3))+np.abs(ndimage.convolve(padded_img,grad_h5)),0) #RB通道的水平梯度
    grad_v=np.where(np.logical_or(Rm==1,Bm==1),np.abs(ndimage.convolve(padded_img,grad_v3))+np.abs(ndimage.convolve(padded_img,grad_v5)),0)
    # 计算R/B通道 H V 的G插值
    Gh=ndimage.convolve(G,filter_h3)+ndimage.convolve(padded_img*(~Gm),filter_h5)    #Gh=(G22+G24)/2+(2R23-R21-R25)/4
    Gv=ndimage.convolve(G,filter_v3)+ndimage.convolve(padded_img*(~Gm),filter_v5)    #Gv=(G13+G33)/2+(2R23-R03-R43)/4
    # 根据梯度计算G
    G=np.where(np.logical_and((grad_h<grad_v),np.logical_or(Rm==1,Bm==1)),Gh,G)
    G=np.where(np.logical_and((grad_v<grad_h),np.logical_or(Rm==1,Bm==1)),Gv,G)
    G=np.where(np.logical_and((grad_v==grad_h),np.logical_or(Rm==1,Bm==1)),(Gh+Gv)/2,G)
    #在B通道插R
    grad_n1=np.array([[1,0,0],
                      [0,0,0],
                      [0,0,-1]])
    grad_n2=np.array([[-1,0,0],
                      [0,2,0],
                      [0,0,-1]])
    grad_p1=np.array([[0,0,1],
                      [0,0,0],
                      [-1,0,0]])
    grad_p2=np.array([[0,0,-1],
                      [0,2,0],
                      [-1,0,0]])
    filter_n1=np.abs(grad_n1)/2
    filter_n2=grad_n2/4
    filter_p1=np.abs(grad_p1)/2
    filter_p2=grad_p2/2
    # grad_nB34=abs(R23-R45)+abs(2G34-G23-G45) grad_pB34=abs(R25-R43)+abs(2G34-G25-G43)
    gradn_at_B=np.abs(ndimage.convolve(R,grad_n1))+np.where((Bm==1),np.abs(ndimage.convolve(G*(Rm+Bm),grad_n2)),0)
    gradp_at_B=np.abs(ndimage.convolve(R,grad_p1))+np.where((Bm==1),np.abs(ndimage.convolve(G*(Rm+Bm),grad_p2)),0)
    # B通道的R插值
    Rn=ndimage.convolve(R,filter_n1)+np.where((Bm==1),ndimage.convolve(G,filter_n2),0) # Rn34=(R23+R45)/2+(2G34-G23-G45)/4
    Rp=ndimage.convolve(R,filter_p1)+np.where((Bm==1),ndimage.convolve(G,filter_p2),0) #Rp34=(R25+R43)/2+(2G34-G25-G43)/4
    #梯度判断
    R=np.where((gradn_at_B<gradp_at_B),Rn,R)
    R=np.where((gradp_at_B<gradn_at_B),Rp,R)
    R=np.where(np.logical_and((gradn_at_B==gradp_at_B),Bm==1),(Rn+Rp)/2,R)
    # 在R通道插值B
    gradn_at_R=np.abs(ndimage.convolve(B,grad_n1))+np.where((Rm==1),np.abs(ndimage.convolve(G*(Rm+Bm),grad_n2)),0)
    gradp_at_R=np.abs(ndimage.convolve(B,grad_p1))+np.where((Rm==1),np.abs(ndimage.convolve(G*(Rm+Bm),grad_p2)),0)
    Bn=ndimage.convolve(B,filter_n1)+np.where((Rm==1),ndimage.convolve(G,filter_n2),0)
    Bp=ndimage.convolve(B,filter_p1)+np.where((Rm==1),ndimage.convolve(G,filter_p2),0)
    B=np.where((gradn_at_R<gradp_at_R),Bn,B)
    B=np.where((gradp_at_R<gradn_at_R),Bp,B)
    B=np.where(np.logical_and((gradn_at_R==gradp_at_R),Rm==1),(Bn+Bp)/2,B)
    #在G通道插值R/B
    R_r=np.transpose(np.any(Rm==1,axis=1)[np.newaxis])*np.ones(R.shape)
    R_c=np.any(Rm==1,axis=0)[np.newaxis]*np.ones(R.shape)
    B_r=np.transpose(np.any(Bm==1,axis=1)[np.newaxis])*np.ones(B.shape)
    B_c=np.any(Bm==1,axis=0)[np.newaxis]*np.ones(B.shape)
    #补齐G通道的R
    R=np.where(np.logical_and(R_r==1,B_c==1),ndimage.convolve(R,filter_h3)+ndimage.convolve(G,filter_h5),R)  #R22=(R21+R23)/2+(2*G22-G21-G23)/4
    R=np.where(np.logical_and(B_r==1,R_c==1),ndimage.convolve(R,filter_v3)+ndimage.convolve(G,filter_v5),R)
    #补齐G通道的B
    B=np.where(np.logical_and(B_r==1,R_c==1),ndimage.convolve(B,filter_h3)+ndimage.convolve(G,filter_h5),B)
    B=np.where(np.logical_and(R_r==1,B_c==1),ndimage.convolve(B,filter_v3)+ndimage.convolve(G,filter_v5),B)
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def mono_demosaic_malvar(bayerimg, bits, bayerpattern):
    bayerimg_=bayerimg.astype(np.int32)
    padded_img=np.pad(bayerimg_,2,mode="reflect")
    Rm,Gm,Bm=mosaic.masks_CFA_bayer(padded_img.shape,bayerpattern)
    R=padded_img*Rm
    G=padded_img*Gm
    B=padded_img*Bm
    #滤波核
    G_at_RB=np.array([[0,0,-1,0,0],
                       [0,0,2,0,0],
                      [-1,2,4,2,-1],
                       [0,0,2,0,0],
                      [0,0,-1,0,0]])/8  #求RB位置的G
    C_at_Gh=np.array([[0,0,0.5,0,0],
                      [0,-1,0,-1,0],
                      [-1,4,5,4,-1],
                      [0,-1,0,-1,0],
                      [0,0,0.5,0,0]])/8 #求邻域在G左右的R和B
    C_at_Gv=np.array([[0,0,-1,0,0],
                      [0,-1,4,-1,0],
                      [0.5,0,5,0,0.5],
                      [0,-1,4,-1,0],
                      [0,0,-1,0,0]])/8 #求邻域在G上下的R和B
    RB_at_BR=np.array([[0,0,-1.5,0,0],
                       [0,2,0,2,0],
                       [-1.5,0,6,0,-1.5],
                       [0,2,0,2,0],
                       [0,0,-1.5,0,0]])/8 #求R和B位置的B和R
    G=np.where(np.logical_or(Rm==1,Bm==1),ndimage.convolve(padded_img,G_at_RB),G)
    RB_at_Gh=ndimage.convolve(padded_img,C_at_Gh)
    RB_at_Gv=ndimage.convolve(padded_img,C_at_Gv)
    RB_BR=ndimage.convolve(padded_img,RB_at_BR)
    #获取R,B所在的行列
    R_r=np.transpose(np.any(Rm==1,axis=1)[np.newaxis])*np.ones(R.shape)
    R_c=np.any(Rm==1,axis=0)[np.newaxis]*np.ones(R.shape)
    B_r=np.transpose(np.any(Bm==1,axis=1)[np.newaxis])*np.ones(B.shape)
    B_c=np.any(Bm==1,axis=0)[np.newaxis]*np.ones(B.shape)
    #求 GR GB位置的R
    R=np.where(np.logical_and(R_r==1,B_c==1),RB_at_Gh,R)
    R=np.where(np.logical_and(B_r==1,R_c==1),RB_at_Gv,R)
    #求GR GB位置的B
    B=np.where(np.logical_and(R_r==1,B_c==1),RB_at_Gv,B)
    B=np.where(np.logical_and(B_r==1,R_c==1),RB_at_Gh,B)
    #求R位置B，B位置R
    R=np.where(np.logical_and(B_r==1,B_c==1),RB_BR,R)
    B=np.where(np.logical_and(R_r==1,R_c==1),RB_BR,B)
    #裁剪膨胀
    result=np.zeros((bayerimg.shape[0],bayerimg.shape[1],3))
    result[:, :, 0] = R[2:-2, 2:-2]
    result[:, :, 1] = G[2:-2, 2:-2]
    result[:, :, 2] = B[2:-2, 2:-2]
    result=np.clip(result,0,2**bits-1)
    return result.astype(bayerimg.dtype)

def demosaic(bayerimg,bits,bayerpattern,mode="malvar"):
    if mode=="bilinear":
        mono_demosaic=mono_demosaic_bilinear
    elif mode=="color_ratio":
        mono_demosaic=mono_demosaic_color_ratio
    elif mode=="color_diff":
        mono_demosaic=mono_demosaic_color_diff
    elif mode=="first_gradient":
        mono_demosaic=mono_demosaic_first_gradient
    elif mode=="second_gradient":
        mono_demosaic=mono_demosaic_second_gradient
    elif mode=="hamilton":
        mono_demosaic=mono_demosaic_hamilton
    elif mode=="malvar":
        mono_demosaic=mono_demosaic_malvar
    else:
        print("Error:demosaic mode is wrong!")
    rgbimg=mono_demosaic(bayerimg,bits,bayerpattern)
    return rgbimg