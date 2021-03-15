import cv2
import numpy as np

def mod(n):
    if(n>0):
        return n
    else:
        return -1*n
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
                    image[i][j] = 255
                else:
                    image[i][j] = 0
        return image

    def calcular_convolucao(wind,filter1):
        val=0
        for i in range(len(filter1)):
            for j in range(len(filter1[0])):
                val+= filter1[i][j]*wind[i][j]
        return val

    def convolucao2(image,filter,n):
        d1 = (int(image.shape[0]/n),int(image.shape[1]/n))
        cv2.resize(image,d1)
        lm = int(n-1-(n-1)/2)
        for i in range(lm,image.shape[0]-lm,2*lm):
            for j in range(lm,image.shape[1]-lm,2*lm):

                for jj in range( -lm,lm+1 ):
                    for ii in range( -lm,lm+1 ):
                        image[i+ii][j+jj] = image[i+ii][j+jj]*filter[ii][jj]

                #convoluted_image[i][j]=OpImage.calcular_convolucao(image,OpImage.janela(image,i,j,n))
        return image
    
    def convolucao(src,filter,n):
        d1 = (int(src.shape[0]/n),int(src.shape[1]/n))
        image = cv2.resize(src,d1)
        lm = int(n-1-(n-1)/2)
        for i in range(lm,image.shape[0]-lm,2*lm):
            for j in range(lm,image.shape[1]-lm,2*lm):
                image[i][j]=OpImage.calcular_convolucao(src,OpImage.janela(src,i,j,n))
        return image

    def rescale(output,min,max):
        min1,max1=0,0
        dist1 = max - min
        for i in output:
            for j in i:
                if j > max1:
                    max1=j
                if j < min1:
                    min1=j
        dist2 = mod(max1) - mod(min1)
        dist = dist1 / dist2
        for i in range(len(output)):
            for j in range(len(output[0])):
                output[i][j]*=dist
                output[i][j]-=min1
        return output
        
        
        


    def convolve(image, kernel):
        # grab the spatial dimensions of the image, along with
        # the spatial dimensions of the kernel
        (iH, iW) = image.shape[:2]
        (kH, kW) = kernel.shape[:2]
        # allocate memory for the output image, taking care to
        # "pad" the borders of the input image so the spatial
        # size (i.e., width and height) are not reduced
        pad = (kW - 1) // 2
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")
        # loop over the input image, "sliding" the kernel across
        # each (x, y)-coordinate from left-to-right and top to
        # bottom
        for y in np.arange(pad, iH + pad):
            for x in np.arange(pad, iW + pad):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (roi * kernel).sum()
                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - pad, x - pad] = k
        # rescale the output image to be in the range [0, 255]
        #output = rescale_intensity(output, in_range=(0, 255))
        output = OpImage.rescale(output, 0,255)
        output = (output * 255).astype("uint8")
        # return the output image
        return output


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