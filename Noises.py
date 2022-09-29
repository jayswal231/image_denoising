from PIL import Image,ImageFilter
import numpy as np
import math
import os
import glob
from math import log10, sqrt
import csv
import pandas as pd
from PIL import ImageDraw 
import matplotlib.pyplot as plt
#original_image= Image.open('/home/mukesh/final_project/pictures/introduction.jpg')
#array_img=np.array(original_image)

#gray_img=original_image.convert('L')
#gray_array=np.array(gray_img)

class Noises:
    
    
   
   
    def GaussianNoise(mean,var,img):
        gauss_noise= np.random.normal(mean,var,img.size)
        gauss_noise=gauss_noise.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
        img_gauss= np.add(img,gauss_noise)
        return img_gauss
    
      

    def Add_sp_noise(img):
        rows, columns, channels = img.shape
        p=0.05
        outputs=np.zeros(img.shape, np.uint8)
        
        for i in range(rows):
            for j in range(columns):
                randoms=np.random.normal(0,1)
                
                if randoms<p/2:
                    outputs[i][j]=0
                elif randoms<p:
                    outputs[i][j]=255
                else:
                    outputs[i][j]= img[i][j]
        return outputs


    def median_filter(img_noisy1):  
        m, n = img_noisy1.shape
        img_new1 = np.zeros([m, n])
         
        for i in range(1, m-1):
            for j in range(1, n-1):
                temp = [img_noisy1[i-1, j-1],
                       img_noisy1[i-1, j],
                       img_noisy1[i-1, j + 1],
                       img_noisy1[i, j-1],
                       img_noisy1[i, j],
                       img_noisy1[i, j + 1],
                       img_noisy1[i + 1, j-1],
                       img_noisy1[i + 1, j],
                       img_noisy1[i + 1, j + 1]]
                 
                temp = sorted(temp)
                img_new1[i, j]= temp[4]
        img_new1 = img_new1.astype(np.uint8)
        
        return img_new1
     
        
    """   
    def adaptive_filter(B):
        M = 5;
        N = 5;
        sz=np.size(B,1)*np.size(B,2)
        #Pad the matrix with zeros on all sides
        C = np.pad(B,[np.math.floor(M/2),np.math.floor(N/2)])



        lvar = np.zeros([np.size(B,1), np.size(B,2)])
        lmean = np.zeros([np.size(B,1), np.size(B,2)])
        temp = np.zeros([np.size(B,1), np.size(B,2)])
        NewImg = np.zeros([np.size(B,1), np.size(B,2)])


        for i in range(1,np.size(C,1)-(M-1)):
            for j in range(1,np.size(C,2)-(N-1)):
                temp = C[(i,i+(M-1)),(j,j+(N-1))]
                tmp =  temp[:]
                lmean(i,j) == tmp.mean(tmp)
                lvar(i,j) == tmp.mean(tmp**2)-tmp.mean(tmp)**2
        nvar = sum(lvar[:])/sz

        #If noise_variance > local_variance then local_variance=noise_variance
        lvar = max(lvar,nvar)     

         #Final_Image = B- (noise variance/local variance)*(B-local_mean)
        NewImg = nvar/lvar
        NewImg = NewImg*(B-lmean)
        NewImg = B-NewImg
        NewImg = NewImg.astype(np.uint8)
        NewImg.show()
        return NewImg            
        """
      
    path= glob.glob('/home/mukesh/final_project/original_images/*.jpg'    )
    img_number =1
    
    
    for file in path:
        original_image=Image.open(file)
        array_img=np.array(original_image)
        gauss_noisy_image=GaussianNoise(0,12,array_img)
#        sp_noisy_image=Add_sp_noise(array_img)
        inputdata=Image.fromarray(gauss_noisy_image)
        inputdata.save('/home/mukesh/final_project/final_images/'+str(img_number)+".jpg")
        img_number +=1
        inputdata.show()
        
        
    def displayImage(pic):
       inputdata=Image.fromarray(pic)
       inputdata.show()
       return inputdata

obj=Noises
original_img=obj.original_image
#sp_img=obj.sp_noisy_image
gauss_img=obj.gauss_noisy_image
#gauss_show=obj.GaussianNoise(0,12,array_img)

#sp_show=obj.Add_sp_noise(array_img)

#mf_show=obj.median_filter(gray_array)
#mdf=obj.medianfilter(sp_show,15)
#adpt_show=obj.adaptive_filter(gauss_show)
#gauss_noisy_img=obj.displayImage(gauss_show)
#sp_noisy_img=obj.displayImage(sp_show)





def bitMix_ordering(pixels,centerRgb,kernelSize) : 

	bitArr = pow(kernelSize,2) * [None]
	p1_24bit = None 

	for i in range(len(pixels)) :
		if pixels[i] == None : 
			p1_24bit = "000000000000000000000000"
		else : 
			red_1 = format(pixels[i][0],'08b')
			green_1 = format(pixels[i][1],'08b')
			blue_1 = format(pixels[i][2],'08b')
			p1_24bit = red_1 + green_1 + blue_1
		bitArr[i] = p1_24bit 

	bitArr.sort()	


	return (int(bitArr[4][:8], 2),int(bitArr[4][8:16], 2),int(bitArr[4][16:24], 2)) 

def lexicographical_ordering(pixels,centerRgb,kernelSize) : # 2 tane parametre alıcak pixel_1 ve pixel_2 
	
	for i in range(len(pixels)) : 
		if pixels[i] == None :  
			pixels[i] = [0,0,0]

	pixels.sort(key=lambda x : x[:])

	return tuple(pixels[4])



def norm_based_ordering(pixels,centerRgb,kernelSize) :  

	EucArr = pow(kernelSize,2) * [None]
	dic = {}
	euc = None
	#print(centerRgb)
	for i in range(len(pixels)) :
		if pixels[i] == None :
			euc = math.sqrt(abs(pow(centerRgb[0],2) + pow(centerRgb[1],2) + pow(centerRgb[2],2)))
			pixels[i] = centerRgb 
		else : 
			euc = math.sqrt(abs(pow((centerRgb[0] - pixels[i][0]),2) + pow((centerRgb[1] - pixels[i][1]),2) + pow((centerRgb[2] - pixels[i][2]),2) ))
		EucArr[i] = euc
		dic[euc] = pixels[i]
#	euc1 = Math.sqrt(Math.pow(firstRed,2) + Math.pow(firstGreen,2) + Math.pow(firstBlue,2));#
#	euc2 = Math.sqrt(Math.pow(secRed,2) + Math.pow(secGreen,2) + Math.pow(secBlue,2));
	EucArr.sort()
	print(EucArr[4])
	print(dic.get(EucArr[4]))

	return tuple(dic.get(EucArr[4]))
    

def scalar_median_filter(image,kernelSize,f) : # marginal

	width, height = image.size

	indexer = kernelSize // 2

	arr_Red = []
	arr_Green = []
	arr_Blue = []
	new_image = image.copy()

	pixel_access_image = new_image.load()


	arr_Red = pow(kernelSize,2) * [None]
	arr_Green = pow(kernelSize,2) * [None]
	arr_Blue = pow(kernelSize,2) * [None]

	for i in range(width) :

		for j in range(height) : 
			y = 0
			for z in range(kernelSize) :
				if i + z - indexer < 0 or i + z - indexer > width - 1:
					for c in range(kernelSize):
						arr_Red[y] = 0
						arr_Green[y] = 0
						arr_Blue[y] = 0	
						y += 1					
				else:
					if j + z - indexer < 0 or j + z - indexer > height - 1:
						arr_Red[y] = 0
						arr_Green[y] = 0
						arr_Blue[y] = 0	
						y += 1
					else:
						for k in range(kernelSize):
							if j + k < height and  i + z < width :
								red, green, blue = image.getpixel((i + z , j + k))
								arr_Red[y] = red
								arr_Green[y] = green
								arr_Blue[y] = blue
							y += 1

			arr_Red.sort()
			arr_Green.sort()
			arr_Blue.sort()
			newRed, newGreen, newBlue = arr_Red[len(arr_Red) // 2],arr_Green[len(arr_Green) // 2], arr_Blue[len(arr_Blue) // 2]
			pixel_access_image[i,j] = (newRed,newGreen,newBlue)

	saveName = "/home/mukesh/final_project/Results/" + f + "_scalar_.png"

	new_image.save(saveName)
	new_image.show()
    	

def vector_median_filter(image,kernelSize,func,f) : # marginal
    width, height = image.size
    indexer = kernelSize // 2

#	arr_Red = []
#	arr_Green = []
#	arr_Blue = []
    new_image = image.copy()

    pixel_access_image = new_image.load()

    sortKernel = pow(kernelSize,2) * [None]

    for i in range(width) :

        for j in range(height) : 
            y = 0
            centerRgb = image.getpixel((i, j))

            for z in range(kernelSize) :
                if i + z - indexer < 0 or i + z - indexer > width - 1:
                    for c in range(kernelSize):
                        sortKernel[y] = None
                        y += 1					
			
                else :
                    if j + z - indexer < 0 or j + z - indexer > height - 1:
                        sortKernel[y] = None
                        y += 1
                    else:
                        for k in range(kernelSize):
                            if j + k < height and  i + z < width :
                                red, green, blue = image.getpixel((i + z , j + k))
                                sortKernel[y] = [red,green,blue]
                                y += 1
            pixel_access_image[i,j] = func(sortKernel,centerRgb,kernelSize)
	
    saveName = "/home/mukesh/final_project/Results" + f + "_vector_" + func.__name__ + ".png"
    new_image.save(saveName)
    name=ImageDraw.Draw(new_image)
    name.text((10,10),"vector median filter", fill=(255,0,0))
    new_image.show()
    return new_image



path =('/home/mukesh/final_project/final_images/')
files = []
#r=root, d=directories, f = files
for r, d, f in os.walk(path):
	for file in f:
		if '.png' or '.jpeg' in file:
			files.append(os.path.join(r, file))

for f in files:
	image = Image.open(f)
	rgb_im = image.convert('RGB')
	vmf=vector_median_filter(rgb_im,3,norm_based_ordering,f.split("/")[1])
#	vector_median_filter(rgb_im,3,bitMix_ordering,f.split("/")[1])
#	vector_median_filter(rgb_im,3,lexicographical_ordering,f.split("/")[1])
#	mmf=scalar_median_filter(rgb_im,3,f.split("/")[1])



#mmf=Image.open("/home/mukesh/final_project/Results/home_scalar_.png")
#sp_img=Image.open("/home/mukesh/final_project/final_images/1.jpg")

def snr(noise_img,filter_img):
    image = np.array(filter_img) ## input orignal image
    mean_image = np.mean(image)
    
    noise_image = np.array(noise_img) ## input noisy image
    noise = noise_image - image
    mean_noise = np.mean(noise)
    noise_diff = noise - mean_noise
    var_noise = np.sum(np.mean(noise_diff**2)) ## variance of noise
    
    if var_noise == 0:
          snr = 100 ## clean image
    else:
          snr = (np.log10(mean_image/var_noise))*20 ## SNR of the image
#          print("SNR:",snr)
      
    return snr      
snr_data=snr(gauss_img,vmf)
#snr_data=snr(gauss_img,mmf)





  
def PSNR(noise_img, filter_img):
    noise_image = np.array(noise_img)
    filter_img = np.array(filter_img)
    mse = np.mean((noise_image - filter_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print("PSNR:"+ str(psnr))
    return psnr
psnr_data=PSNR(gauss_img,vmf)
#psnr_data=PSNR(gauss_img,mmf)

with  open('/home/mukesh/final_project/Results/data/data.csv', 'w',newline='') as file:
          writer=csv.writer(file)
#              writer.writerow(["SN", "NAME", "SNR", "PSNR"])
          fil=writer.writerow(["FOR VECTOR MEDIAN:" + "SNR:" + str(snr_data)+ " " + "PSNR:" + str(psnr_data)+ " " + "db"])
#          writer.writerow(["FOR MARGINAL MEDIAN:" + "SNR:" + str(snr_data)+ " " + "PSNR:" + str(psnr_data) + " " + "db"])
         
data=pd.read_csv('/home/mukesh/final_project/Results/data/data.csv')
print(data)
          #fil.close()
#         fil.write()




   
