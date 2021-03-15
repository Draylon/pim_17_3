import cv2
import numpy as np
from Op import OpImage

image = cv2.imread("./images/casa.png")

#d1=( image.shape[0]*3,image.shape[1]*2 )
#image = cv2.resize(image,d1)

#l1 = image.janela(image.getMidWidth(),image.getMidHeight(),3)

gray = OpImage.grayscale(image)

#thold50 = OpImage.threshold(grayscale.copy(),200)

#f1 = [[-1,0,1],[-2,0,2],[-1,0,1]]
#f1 = [[-1,-2,-1],[0,0,0],[1,2,1]]
#f1 = [[0,1,0],[0,0,0],[0,0,0]]
#f1 = [[0,1,0],[1,0,1],[0,1,0]]
#f1 = [[-1,-1,0],[-1,8,1],[-1,-1,-1]]

#conv = OpImage.convolucao(thold50.copy(),f1,3)


#============

#cv2.imshow('original',image)
#cv2.imshow('grayscale',grayscale)
#cv2.imshow('Threshold',thold50)
#cv2.imshow('convoluted',conv)

#cv2.waitKey(0)

#cv2.destroyAllWindows()



'''
Parte A: implemente uma função “janela deslizante” que a partir da localização (i,j) 
do centro de  uma  janela  NxN  (N  ímpar),  retorna  as  coordenadas  dos  vizinhos 
de  (i,j). Inicialmente faça para N=3.

Parte BPara a imagem img02.jpg disponível nas imagens para testes (Moodle), implemente 
o filtro de reforço utilizando o operador de média ao invés do gaussiano e a função de 
posicionamento de janela deslizante que vc implementou na Parte-A, anterior

'''

# construct the argument parse and parse the arguments
# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")
# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")
# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")


# construct the kernel bank, a list of kernels we're going
# to apply using both our custom `convole` function and
# OpenCV's `filter2D` function
kernelBank = (
	("small_blur", smallBlur),
	("large_blur", largeBlur),
	("sharpen", sharpen),
	("laplacian", laplacian),
	("sobel_x", sobelX),
	("sobel_y", sobelY)
)


# load the input image and convert it to grayscale


# loop over the kernels
for (kernelName, kernel) in kernelBank:
	# apply the kernel to the grayscale image using both
	# our custom `convole` function and OpenCV's `filter2D`
	# function
	print("[INFO] applying {} kernel".format(kernelName))
	convoleOutput = OpImage.convolve(gray, kernel)
	opencvOutput = cv2.filter2D(gray, -1, kernel)
	# show the output images
	cv2.imshow("original", gray)
	cv2.imshow("{} - convole".format(kernelName), convoleOutput)
	cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
	cv2.waitKey(0)
	cv2.destroyAllWindows()