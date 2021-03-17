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

    def threshold(image,threshold,inverse=False):
        out = image.copy()
        match = 255 if inverse == False else 0
        no_match = 0 if inverse == False else 255
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j] >= threshold:
                    out[i][j] = match
                else:
                    out[i][j] = no_match
        return out

    def rescale(image,min,max):
        externo_min, externo_max = 0,0
        if(min>=0):
            externo_min=0
            externo_max = 1

        #image = np.clip(image, min, max)
        for i in range(len(image)):
            for j in range(len(image[0])):
                if image[i][j] > max:
                    image[i][j] = max
                elif image[i][j] < min:
                    image[i][j] = min
                if min != max:
                    image[i][j] = (image[i][j] - min) / (max - min)
            
        if min != max:
            return np.asarray(image * (externo_max - externo_min) + externo_min, dtype=image.dtype.type)
        else:
            return np.clip(image, externo_min, externo_max).astype(image.dtype.type)
        
        
        

    #https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    #https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    def convolve(image, kernel):
        # grab the spatial dimensions of the image, along with
        # the spatial dimensions of the kernel
        (iH, iW) = image.shape[0],image.shape[1]
        (kH, kW) = kernel.shape[0],kernel.shape[1]
        # allocate memory for the output image, taking care to
        # "pad" the borders of the input image so the spatial
        # size (i.e., width and height) are not reduced
        kernel_padding = int((kW - 1) / 2)
        image = cv2.copyMakeBorder(image, kernel_padding, kernel_padding, kernel_padding, kernel_padding, cv2.BORDER_REPLICATE)
        output = np.zeros((iH, iW), dtype="float32")
        # loop over the input image, "sliding" the kernel across
        # each (x, y)-coordinate from left-to-right and top to
        # bottom
        for y in np.arange(kernel_padding, iH + kernel_padding):
            for x in np.arange(kernel_padding, iW + kernel_padding):
                # extract the ROI of the image by extracting the
                # *center* region of the current (x, y)-coordinates
                # dimensions
                roi = image[y - kernel_padding:y + kernel_padding + 1, x - kernel_padding:x + kernel_padding + 1]
                # perform the actual convolution by taking the
                # element-wise multiplicate between the ROI and
                # the kernel, then summing the matrix
                k = (roi * kernel).sum()
                # store the convolved value in the output (x,y)-
                # coordinate of the output image
                output[y - kernel_padding, x - kernel_padding] = k
        # rescale the output image to be in the range [0, 255]
        #output = rescale_intensity(output, in_range=(0, 255))
        output = OpImage.rescale(output, 0,255)
        output = (output * 255).astype("uint8")
        # return the output image
        return output

    def media_convolucao(image,n):
        media_matriz = np.ones((n,n)) * (1 / n**2)
        return OpImage.convolve(image,media_matriz)



    def median(image,n): # pegando os cantos
        out = image.copy()
        marg = int((n-1)/2)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                med_ = 0
                ctx = 0
                for wi in range(-marg,marg+1):
                    for wj in range(-marg,marg+1):
                        if x-wi < 0 or y-wj < 0 or x-wi >= image.shape[0] or y-wj >= image.shape[1]:
                            continue
                        if wi == wj and wi == marg:
                            continue
                        med_+= image[x-wi][y-wj]
                        ctx+=1
                med_/=ctx
                out[x][y]=med_
        return out


    def median_margin(image,n):
        #window = np.zeros((n,n))
        out = image.copy()
        marg = int((n-1)/2)

        for x in range(marg,image.shape[0]-marg):
            for y in range(marg,image.shape[1]-marg):
                ii=0
                for i in range(n):
                    for j in range(n):
                        if not(i == j and i == marg):
                            ii += image[x + i - marg][y + j - marg]
                ii/=(n*n)-1

                #sort entries in window[]
                out[x][y] = ii #window[mid_][mid_]
        return out
        



    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''cv2.rectangle(None,(0,0),(image.shape[0],image.shape[1]),0)
        image.shape = (image.shape[0],image.shape[1],1)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                a1,a2,a3 = image[i][j]
                image[i][j] = 0.299*a1 + 0.587*a2 + 0.114*a3'''


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