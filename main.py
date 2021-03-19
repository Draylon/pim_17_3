import cv2
import numpy as np
from Op import OpImage
from kernels import *
import matplotlib.pyplot
#from hog_ex import *

image1 = cv2.imread("./images/carro1.jpg")
#gray = cv2.imread("./images/squaretfu.jpg")
#image1 = np.float32(image1) / 255.0
#d1=( image.shape[0]*3,image.shape[1]*2 )
#image = cv2.resize(image,d1)

#l1 = image.janela(image.getMidWidth(),image.getMidHeight(),3)

gray = OpImage.grayscale(image1)
sx = OpImage.convolucao(gray,sobelX)
sy = OpImage.convolucao(gray,sobelY)
magnitude,angulos = OpImage.mag_direction(sx,sy)
magnitude_limiar = np.std(magnitude)
print( type(magnitude_limiar) ,magnitude_limiar)
limiar1 = OpImage.threshold_media(magnitude,[0,1,2],magnitude_limiar)

#gray = gray - OpImage.media_conectividade(gray,5,(0,100,True))
#gray = OpImage.convolucao(gray,sharpen)

'''gray = OpImage.media_convolucao(gray,5)
gray = OpImage.convolucao(gray,unsharpmasking)
sx = OpImage.convolucao(gray,sobelX)
sy = OpImage.convolucao(gray,sobelY)
ss = OpImage.merge_max(sx,sy)

mag,ang = OpImage.mag_direction(sx,sy)

#_,visualize = build_histogram(mag,ang,(8,8),False,9,(3,3),True,False,True)

hog_data = OpImage.hog(mag,ang)
hog_image = OpImage.drawHOG(hog_data)
#hog_image = OpImage.threshold(hog_image,0.95)

hog_original = OpImage.rgbSum(image1,ss)

'''

cv2.imshow("gray",gray)
cv2.imshow("sx",sx)
cv2.imshow("sy",sy)
#cv2.imshow("SS",ss)
#cv2.imshow("HOG IMAGE",hog_image)
#cv2.imshow("HOG OFFICIAL",visualize)
#cv2.imshow("IMAGEM ORIGINAL",image1)
#cv2.imshow("HOG OVERLAY",hog_original)

cv2.waitKey(0)
cv2.destroyAllWindows()

#thold50 = OpImage.threshold(grayscale.copy(),200)

#runAllKernels(gray)


'''
Parte A: implemente uma função “janela deslizante” que a partir da localização (i,j) 
do centro de  uma  janela  NxN  (N  ímpar),  retorna  as  coordenadas  dos  vizinhos 
de  (i,j). Inicialmente faça para N=3.

Parte BPara a imagem img02.jpg disponível nas imagens para testes (Moodle), implemente 
o filtro de reforço utilizando o operador de média ao invés do gaussiano e a função de 
posicionamento de janela deslizante que vc implementou na Parte-A, anterior

'''

