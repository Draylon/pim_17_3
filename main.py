import cv2
from Op import OpImage

image1 = cv2.imread("./images/casa.png")

#l1 = image1.janela(image1.getMidWidth(),image1.getMidHeight(),3)

grayscale = OpImage.grayscale(image1)

thold50 = OpImage.threshold(grayscale,50)


f1 = [[-1,0,1],[-2,0,2],[-1,0,1]]
#f1 = [[-1,-2,-1],[0,0,0],[1,2,1]]
#f1 = [[0,1,0],[0,0,0],[0,0,0]]
#f1 = [[0,1,0],[1,0,1],[0,1,0]]
#f1 = [[-1,-1,0],[-1,8,1],[-1,-1,-1]]

conv = OpImage.convolucao(thold50,f1,3)


#============

cv2.imshow('original',image1)
cv2.imshow('grayscale',grayscale)
cv2.imshow('original',thold50)
cv2.imshow('convoluted',conv)

cv2.waitKey(0)


'''

Parte A: implemente uma função “janela deslizante” que a partir da localização (i,j) 
do centro de  uma  janela  NxN  (N  ímpar),  retorna  as  coordenadas  dos  vizinhos 
de  (i,j). Inicialmente faça para N=3.

Parte BPara a imagem img02.jpg disponível nas imagens para testes (Moodle), implemente 
o filtro de reforço utilizando o operador de média ao invés do gaussiano e a função de 
posicionamento de janela deslizante que vc implementou na Parte-A, anterior

'''

