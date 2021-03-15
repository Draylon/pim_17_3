import cv2
import numpy as np
class OpImage:
    
    def janela(image,x,y,n):
        #if n % 2 == 0: raise Exception("'N' deve ser Ímpar")
        ret = []
        lm = int(n-1-(n-1)/2)
        for j in range( -lm,lm+1 ):
            ret.append([])
            for i in range( -lm,lm+1 ):
                ret[-1].append(image[x+i][y+j])
        return ret

    def threshold(image,threshold):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= threshold:
                    image[i][j] = 1
                else:
                    image[i][j] = 0
        return image

    def calcular_convolucao(wind,filter1):
        val=0
        for i in range(len(filter1)):
            for j in range(len(filter1[0])):
                val+= filter1[i][j]*wind[i][j]
        return val

    def convolucao(image,filter,n):
        lm = int(n-1-(n-1)/2)
        for i in range(lm,image.shape[0]-lm,2*lm):
            for j in range(lm,image.shape[1]-lm,2*lm):

                for jj in range( -lm,lm+1 ):
                    for ii in range( -lm,lm+1 ):
                        image[i+ii][j+jj] = image[i+ii][j+jj]*filter[ii][jj]

                #convoluted_image[i][j]=OpImage.calcular_convolucao(image,OpImage.janela(image,i,j,n))
        return image


    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''cv2.rectangle(None,(0,0),(image.shape[0],image.shape[1]),0)
        image.shape = (image.shape[0],image.shape[1],1)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                a1,a2,a3 = image[i][j]
                image[i][j] = 0.299*a1 + 0.587*a2 + 0.114*a3'''
                

    def wait(self):
        cv2.waitKey(0)


    '''        return
    ([
        image1[x-1][y-1],
        image1[x][y-1],
        image1[x+1][y-1]
    ],[
        image1[x-1][y],
        image1[x][y],
        image1[x+1][y]
    ],[
        image1[x-1][y+1],
        image1[x][y+1],
        image1[x+1][y+1]
    ])'''