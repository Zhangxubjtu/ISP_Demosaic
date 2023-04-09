import os,time
import cv2 as cv
import pandas as pd
import src.mosaicing as mosaicing
import src.demosaicing as demosaicing
import src.imquality as imquality

if __name__=="__main__":
    dataset=r'data\Kodak Lossless True Color Image Dataset'
    bits=8
    bayerpattern="RGGB"
    outpath=".\Output"
    os.makedirs(outpath,exist_ok=True)
    xlspath = ".\Output\demosaic_quality.xlsx"
    psnr,ssim={},{}
    for file in os.listdir(dataset):
        filepath=os.path.join(dataset,file)
        img=cv.imread(filepath)
        #mosaic
        rgbimg = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        bayerimg=mosaicing.mosaicing_CFA_bayer(rgbimg,bayerpattern)
        cv.imwrite(os.path.join(outpath,file.split('.')[0]+"_0_original"+".png"),img)
        cv.imwrite(os.path.join(outpath, file.split('.')[0] + "_1_mosaic" + ".png"), bayerimg)
        #demosaic所有方法遍历
        modes=["bilinear","color_ratio","color_diff","first_gradient","second_gradient","hamilton","malvar"]
        psnr_list,ssim_list=[],[]
        modes_psnr,modes_ssim={},{}
        for i,mode in enumerate(modes):
            start = time.time()
            demosaic_img=demosaicing.demosaic(bayerimg,bits,bayerpattern,mode=mode)
            print(file+" demosaic mode: "+mode+" cost ",time.time()-start," s.")
            demosaic_img=cv.cvtColor(demosaic_img, cv.COLOR_RGB2BGR)
            mode_psnr, mode_ssim = imquality.imquality_test(img, demosaic_img, bits)
            psnr_list.append(mode_psnr)
            ssim_list.append(mode_ssim)
            cv.imwrite(os.path.join(outpath, file.split('.')[0] + "_"+str(i+2)+"_"+mode + ".png"), demosaic_img)
        for i in range(len(psnr_list)):
            modes_psnr[modes[i]] = psnr_list[i]
            modes_ssim[modes[i]] = ssim_list[i]
        psnr[file.split('.')[0]]=modes_psnr
        ssim[file.split('.')[0]]=modes_ssim
    #将psnr,ssim数据保存到excel
    df_psnr=pd.DataFrame(psnr).T
    df_ssim=pd.DataFrame(ssim).T
    with pd.ExcelWriter(xlspath) as writer:
        df_psnr.to_excel(writer,sheet_name="PSNR")
        df_ssim.to_excel(writer,sheet_name="SSIM")