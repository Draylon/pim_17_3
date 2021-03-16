import cv2
import numpy as np
from Op import OpImage
from kernels import *

image = cv2.imread("./images/casa.png")

#d1=( image.shape[0]*3,image.shape[1]*2 )
#image = cv2.resize(image,d1)

#l1 = image.janela(image.getMidWidth(),image.getMidHeight(),3)

gray = OpImage.grayscale(image)

#thold50 = OpImage.threshold(grayscale.copy(),200)

runAllKernels(gray)


'''
Parte A: implemente uma função “janela deslizante” que a partir da localização (i,j) 
do centro de  uma  janela  NxN  (N  ímpar),  retorna  as  coordenadas  dos  vizinhos 
de  (i,j). Inicialmente faça para N=3.

Parte BPara a imagem img02.jpg disponível nas imagens para testes (Moodle), implemente 
o filtro de reforço utilizando o operador de média ao invés do gaussiano e a função de 
posicionamento de janela deslizante que vc implementou na Parte-A, anterior

'''

