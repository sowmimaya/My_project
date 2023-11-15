import cv2
import numpy as np

from tkinter import *
from PIL import ImageTk,Image
from os.path import join
from os import listdir
saving_directory=r"D:\Mini project\Main project"
browsing_directory=r"D:\Mini project\Main project"

images=Image.open('bamboo' + '.jpg')
image_array=[]
count=0
for image in images:
	im=cv2.imread(join(browsing_directory,image))
	exten="."+image.split(".")[-1]
	image_array.append(im)
	print(f"Image{image}has dimensions as {np.shape(im)[0]}X{np.shape(im)[1]}.")
	x,y,w,h=cv2.selectROI(im)
	cropped_image=im[y:y+h,x:x+w]
	cv2.imshow("Res",cropped_image)
	print(f"Image created status: {cv2.imwrite(join(saving_directory,str(count)+exten),cropped_image)}")
	count=count+1
	print("process complete....")
cv2.destroyAllWindows()	