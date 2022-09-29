
from PIL import Image,ImageFilter
import numpy as np
import csv
"""
import glob
path= glob.glob('/home/mukesh/final_project/original_images/*.jpg'    )
img_number =1


for file in path:
    original_image=Image.open(file)
    gray_image=original_image.convert('L')
    noisy_image = gray_image.filter(ImageFilter.GaussianBlur(5))
    noisy_image.save('/home/mukesh/final_project/final_images/'+str(img_number)+".jpg")
    img_number +=1
    noisy_image.show()
    
"""    
    
"""    
def signalTonoise():
    signal=np.square(array_img)
    sum_signal=np.sum(signal,dtype=np.int64)
    noise=np.square(gauss_show)
    sum_noise=np.sum(noise, dtype=np.int64)
    snr=sum_signal/sum_noise
    print(snr)
    return snr
    
signalTonoise()

from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(noise_img, filter_img):
    mse = np.mean((noise_img - filter_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     noise_img = cv2.imread("/home/mukesh/final_project/pictures/b.png")
     filter_img = cv2.imread("/home/mukesh/final_project/pictures/lovely.jpg", 1)
     value = PSNR(noise_img, filter_img)
     print(f"PSNR value is {value} dB")
       
if __name__ == "__main__":
    main()
 """
filter_img=Image.open("/home/mukesh/final_project/Results/home_scalar_.png")
noise_img=Image.open("/home/mukesh/final_project/final_images/1.jpg")

def snr(noise_img,filter_img):
    image = np.array(filter_img) ## input orignal image
    mean_image = np.mean(image)
    
    noisy_image = np.array(noise_img) ## input noisy image
    noise = noisy_image - image
    mean_noise = np.mean(noise)
    noise_diff = noise - mean_noise
    var_noise = np.sum(np.mean(noise_diff**2)) ## variance of noise
    
    if var_noise == 0:
          snr = 100 ## clean image
    else:
          snr = (np.log10(mean_image/var_noise))*20 ## SNR of the image
          print(snr)
#          with  open('/home/mukesh/final_project/Results/data/data.csv', 'w',newline='') as file:
#              writer=csv.writer(file)
#              writer.writerow(["SN", "NAME", "SNR", "PSNR"])
#              writer.writerow(["SNR FOR VECTOR MEDIAN:" + str(snr)])
#              fil.close()
#              fil.write("SNR: " + str(snr) + " db")
         

    return snr      
snr_data=snr(noise_img,filter_img)

from math import log10, sqrt
import cv2

  
def PSNR(noise_img, filter_img):
    noise_img = np.array(noise_img)
    filter_img = np.array(filter_img)
    mse = np.mean((noise_img - filter_img) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    print(psnr)
    return psnr
psnr_data=PSNR(noise_img,filter_img)

with  open('/home/mukesh/final_project/Results/data/data.txt', 'w',newline='') as file:
          writer=csv.writer(file)
#              writer.writerow(["SN", "NAME", "SNR", "PSNR"])
          writer.writerows(["FOR VECTOR MEDIAN:" + "SNR:" + str(snr_data)+ " " + "PSNR:" + str(psnr_data)])
#              fil.close()
#              fil.write("SNR: " + str(snr) + " db")




"""  
def main():
     noise_img = cv2.imread("noise_img.png")
     filter_img = cv2.imread("filter_img.png", 1)
     value = PSNR(noise_img, filter_img)
     print(f"PSNR value is {value} dB")
     fil = open('PSNR.txt', 'w')
     fil.write(" PSNR: " + str(value) + " db") 
if __name__ == "__main__":
    main()
    """