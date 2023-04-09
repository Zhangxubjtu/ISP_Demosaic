import numpy as np

def imquality_psnr_onechannel(img1,img2,bits):
    """
    PSNR:Peak Signal to Noise Ratio=10*log10(L^2/MSE)
    MSE:Mean Square Error=((img1-img2)^2).mean()
    :param img1:
    :param img2:
    :param bits:
    :return:
    """
    img1_=img1.astype(np.int32)
    img2_=img2.astype(np.int32)
    mse=((img1_-img2_)**2).mean()
    if mse==0:
        psnr=100
    else:
        L=2**bits-1
        psnr=10*np.log10(L**2/mse)
    return psnr

def imquality_ssim_onechannel(img1,img2,bits):
    """
    ssim:Structural Similarity=Light(X,Y)*Contrast(X,Y)*Structural(X,Y)
    L(X,Y)=(2*ux*uy+C1)/(ux^2+uy^2+C1)
    C(X,Y)=(2*sigmax*sigmay+C2)/(sigmax^2+sigmay^2+C2)
    S(X,Y)=(sigmaxy+C3)/(sigmax*sigmay+C3)
    C1=(k1*L)^2,C2=(k2*L)^2,C3=C2/2
    :param img1:
    :param img2:
    :param bits:
    :return:[0,1]，1=similar
    """
    img1_=img1.astype(np.int32)
    img2_=img2.astype(np.int32)
    mu1=img1_.mean()
    mu2=img2_.mean()
    sigma1=np.sqrt(((img1_-mu1)**2).mean())
    sigma2=np.sqrt(((img2_-mu2)**2).mean())
    sigma12=((img1_-mu1)*(img2_-mu2)).mean()
    L=2**bits-1
    k1=0.01
    k2=0.03
    C1=(k1*L)**2
    C2=(k2*L)**2
    C3=C2/2
    Light=(2*mu1*mu2+C1)/(mu1**2+mu2**2)
    Contrast=(2*sigma1*sigma2+C2)/(sigma1**2+sigma2**2+C2)
    Structural=(sigma12+C3)/(sigma1*sigma2+C3)
    ssim=Light*Contrast*Structural
    return ssim

def imquality_psnr(img1,img2,bits,title=None):
    assert img1.shape==img2.shape
    if len(img1.shape)==2:
        psnr=imquality_psnr_onechannel(img1,img2,bits)
    elif len(img1.shape)==3:
        psnr=np.zeros(4,dtype=np.float32)
        for i in range(3):
            psnr[i]=imquality_psnr_onechannel(img1[:,:,i],img2[:,:,i],bits)
        psnr[3]=psnr[:3].mean()
    if title is not None:
        print(title,"PSNR:",psnr)
    return psnr

def imquality_ssim(img1,img2,bits,title=None):
    assert img1.shape==img2.shape
    if len(img1.shape)==2:
        ssim=imquality_ssim_onechannel(img1,img2,bits)
    elif len(img1.shape)==3:
        ssim=np.zeros(4,dtype=np.float32)
        for i in range(3):
            ssim[i]=imquality_ssim_onechannel(img1[:,:,i],img2[:,:,i],bits)
        ssim[3]=ssim[:3].mean()
    if title is not None:
        print(title,'SSIM:',ssim)
    return ssim

def imquality_test(img1,img2,bits,title=None):
    psnr=imquality_psnr(img1,img2,bits,title)
    ssim=imquality_ssim(img1,img2,bits,title)
    return psnr,ssim