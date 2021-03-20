import cv2
import numpy as np
from Op import OpImage
from kernels import *
import matplotlib.pyplot as plt
#from hog_ex import *

image1 = cv2.imread("./images/img02.jpg")
#gray = cv2.imread("./images/squaretfu.jpg")
#image1 = np.float32(image1) / 255.0
#d1=( image.shape[0]*3,image.shape[1]*2 )
#image = cv2.resize(image,d1)

#l1 = image.janela(image.getMidWidth(),image.getMidHeight(),3)
gray = OpImage.grayscale(image1)
sx = OpImage.convolucao(gray,sobelX)
sy = OpImage.convolucao(gray,sobelY)
ss = OpImage.sobelRgb(sx,sy)

magnitude,angulos = OpImage.mag_direction(sx,sy)
limiar1 = OpImage.threshold_media(magnitude,[1.7])

mag,ang = OpImage.mag_direction(sx,sy)

#_,visualize = build_histogram(mag,ang,(8,8),False,9,(3,3),True,False,True)


hog_dataC = [ OpImage.hog(xx,angulos) for xx in limiar1 ]
#hog_imageC = [ OpImage.drawHOG(hcd) for hcd in hog_dataC ]

'''for hogmg in hog_imageC:
    med=0
    for i in hogmg:
        for j in i:
            med+=j
    print("media",med)'''

#hog_image = OpImage.threshold(hog_image,0.95)
#hog_original = OpImage.rgbSum(image1,ss)

hog_hist = OpImage.agrupar_hog(hog_dataC[0],57,24,4)
OpImage.drawHOGHistogram(hog_hist)


cv2.imshow("gray",gray)
cv2.imshow("sx",sx)
cv2.imshow("sy",sy)
#for ind,img in enumerate(hog_imageC):
#    cv2.imshow("HOG IMAGE"+str(ind),img)
#cv2.imshow("HOG OFFICIAL",visualize)
cv2.imshow("IMAGEM ORIGINAL",image1)
cv2.imshow("SOBEL",ss)
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

