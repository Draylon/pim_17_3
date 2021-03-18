import cv2
import numpy as np
from Op import OpImage
from kernels import *
import matplotlib.pyplot

image1 = cv2.imread("./images/carro1.jpg")
#gray = cv2.imread("./images/squaretfu.jpg")
#image1 = np.float32(image1) / 255.0
#d1=( image.shape[0]*3,image.shape[1]*2 )
#image = cv2.resize(image,d1)

#l1 = image.janela(image.getMidWidth(),image.getMidHeight(),3)

gray = OpImage.grayscale(image1)

sx = OpImage.convolucao(gray,sobelX)
sy = OpImage.convolucao(gray,sobelY)

#mag,ang = OpImage.mag_direction(sx,sy)
mag, ang = cv2.cartToPolar(sx, sy, angleInDegrees=True)

OpImage.hog(mag,ang)

#grad2 = OpImage.calculate_gradient(gray,np.array([-1,0,1]))
#med1=OpImage.gradienteMedia(grad)
#thd_list = OpImage.threshold_media(grad,[0,1,2],10)
#sc = sx + sy
#sc = OpImage.merge_max(sx,sy)

cv2.imshow("gray",gray)
cv2.imshow("sx",sx)
cv2.imshow("sy",sy)
#cv2.imshow("mag1",mag)
#cv2.imshow("angle",ang)
#cv2.imshow("sc",sc)
#cv2.imshow("grad",grad)

#for thdi,thd in enumerate(thd_list):
#    cv2.imshow("thold"+str(thdi),thd)

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

