from PIL import Image
import math
import os



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
	new_image.show()


def main() : 
	path =('/home/mukesh/final_project/pictures/')
    
	files = []
#	r=root, d=directories, f = files
	for r, d, f in os.walk(path):
		for file in f:
			if '.png' or '.jpeg' in file:
				files.append(os.path.join(r, file))

	for f in files:
		image = Image.open(f)
		rgb_im = image.convert('RGB')
		vector_median_filter(rgb_im,3,norm_based_ordering,f.split("/")[1])
#		vector_median_filter(rgb_im,3,bitMix_ordering,f.split("/")[1])
#		vector_median_filter(rgb_im,3,lexicographical_ordering,f.split("/")[1])
		scalar_median_filter(rgb_im,3,f.split("/")[1])


if __name__ == "__main__" : 
	main()












