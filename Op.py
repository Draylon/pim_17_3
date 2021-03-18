import cv2
import numpy as np
from math import sqrt,atan2
import matplotlib.pyplot as plt

def mod(n):
    if(n>0):
        return n
    else:
        return -1*n
class OpImage:
    
    
                
    @staticmethod
    def janela(image,x,y,n):
        #if n % 2 == 0: raise Exception("'N' deve ser Ímpar")
        ret = []
        lm = int(n-1-(n-1)/2)
        for j in range( -lm,lm+1 ):
            ret.append([])
            for i in range( -lm,lm+1 ):
                ret[-1].append(image[x+i][y+j])
        return ret

    @staticmethod
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

    @staticmethod
    def rescale(image,min,max):
        externo_min, externo_max = 0,0
        if(min>=0):
            externo_min=0
            externo_max = 1
        image = np.clip(image, min, max)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
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
        
    @staticmethod
    def mag_direction(im1,im2):
        mag = np.ones((im1.shape[0],im1.shape[1]))
        dir1 = np.ones((im1.shape[0],im1.shape[1]))
        for x in range(im1.shape[0]):
            for y in range(im1.shape[1]):
                mag[x][y] = sqrt( (im1[x][y]**2 + im2[x][y] ** 2) )
                dir1[x][y] = atan2( im2[x][y],im1[x][y] )
        dir1 = np.rad2deg(dir1)
        dir1 = dir1%180
        mag = OpImage.rescale(mag, 0,255)
        mag = (mag * 255).astype("uint8")
        return mag,dir1

    @staticmethod
    def threshold_media(image,k,sigma):
        out=[image.copy()]*len(k)
        rn = range(len(k))
        T = [OpImage.gradienteMedia(image) - kk*sigma for kk in k]
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                for ti in rn:
                    if image[x][y] >= T[ti]:
                        out[ti][x][y]=255
                    else:
                        out[ti][x][y]=0
        return out

    '''𝐼𝑠𝑎í𝑑𝑎(𝑥,𝑦)=255 𝑠𝑒 𝑚𝑎𝑔(𝑥,𝑦)≥𝑇
    𝐼𝑠𝑎í𝑑𝑎(𝑥,𝑦)=0 𝑠𝑒 𝑚𝑎𝑔(𝑥,𝑦)<𝑇'''

    '''@staticmethod
    def calculate_gradient(img, template):
        ts = template.size #Number of elements in the template (3).
        #New padded array to hold the resultant gradient image.
        new_img = np.zeros((img.shape[0]+ts-1, 
                            img.shape[1]+ts-1))
        new_img[np.uint16((ts-1)/2.0):img.shape[0]+np.uint16((ts-1)/2.0), 
                np.uint16((ts-1)/2.0):img.shape[1]+np.uint16((ts-1)/2.0)] = img
        result = np.zeros((new_img.shape))
        
        for r in np.uint16(np.arange((ts-1)/2.0, img.shape[0]+(ts-1)/2.0)):
            for c in np.uint16(np.arange((ts-1)/2.0, 
                                img.shape[1]+(ts-1)/2.0)):
                curr_region = new_img[r-np.uint16((ts-1)/2.0):r+np.uint16((ts-1)/2.0)+1, 
                                    c-np.uint16((ts-1)/2.0):c+np.uint16((ts-1)/2.0)+1]
                curr_result = curr_region * template
                score = np.sum(curr_result)
                result[r, c] = score
        #Result of the same size as the original image after removing the padding.
        result_img = result[np.uint16((ts-1)/2.0):result.shape[0]-np.uint16((ts-1)/2.0), 
                            np.uint16((ts-1)/2.0):result.shape[1]-np.uint16((ts-1)/2.0)]
        return result_img'''

    @staticmethod
    def gradienteMedia(grad1):
        med = 0
        for x in range(grad1.shape[0]):
            for y in range(grad1.shape[1]):
                med+=grad1[x][y]
        med/=grad1.shape[0]*grad1.shape[1]
        return med

    #https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    #https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    @staticmethod
    def convolucao(image, kernel):
        (iW, iH) = image.shape[0],image.shape[1]
        (kW, kH) = kernel.shape[0],kernel.shape[1]
        output = np.full((iW, iH),128, dtype="float32")
        kernel_padding = int((kW - 1) / 2)
        image = cv2.copyMakeBorder(image, kernel_padding, kernel_padding, kernel_padding, kernel_padding, cv2.BORDER_REPLICATE)
        for x in np.arange(kernel_padding, iW + kernel_padding):
            for y in np.arange(kernel_padding, iH + kernel_padding):
                ROI = image[x - kernel_padding:x + kernel_padding + 1, y - kernel_padding:y + kernel_padding + 1]
                k = (ROI * kernel).sum()
                output[x - kernel_padding, y - kernel_padding] = mod(k)
        output = OpImage.rescale(output, 0,255)
        output = (output * 255).astype("uint8")
        return output

    @staticmethod
    def media_convolucao(image,n):
        media_matriz = np.ones((n,n)) * (1 / n**2)
        return OpImage.convolve(image,media_matriz)

    @staticmethod
    def media(image,n): # pegando os cantos
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

    @staticmethod
    def media_overwrite(image,n): # sobrescrevendo na imagem
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
                        med_+= out[x-wi][y-wj]
                        ctx+=1
                med_/=ctx
                out[x][y]=med_
        return out

    @staticmethod
    def media_nomargin(image,n): # não pega os cantos
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
        
    @staticmethod
    def media_conectividade(image,n,in_range=(0,255),inverse=False):
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

                if mod(image[x][y] - med_) > in_range[0] and mod(image[x][y] - med_) <= in_range[1]:
                    out[x][y] = 0
                '''if not inverse:
                    if mod(image[x][y] - med_) > in_range[0] and mod(image[x][y] - med_) < in_range[1]:
                        out[x][y] = 0
                else:
                    if (mod(image[x][y] - med_)) < in_range[0] and (mod(image[x][y] - med_)) > in_range[0]:
                        out[x][y] = 0'''
                
        return out

    @staticmethod
    def merge_min(image,subtr):
        nshape = image.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if subtr[x][y] >= image[x][y]:
                    nshape[x][y]=0
                else:
                    nshape[x][y]=image[x][y]-subtr[x][y]
        return nshape

    @staticmethod
    def merge_max(image,add):
        nshape = image.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                if add[x][y] >= image[x][y]:
                    nshape[x][y]=add[x][y]
                else:
                    nshape[x][y]=image[x][y]
        return nshape


    @staticmethod
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        '''cv2.rectangle(None,(0,0),(image.shape[0],image.shape[1]),0)
        image.shape = (image.shape[0],image.shape[1],1)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                a1,a2,a3 = image[i][j]
                image[i][j] = 0.299*a1 + 0.587*a2 + 0.114*a3'''

#=======================

    @staticmethod
    def HOG_cell_histogram(cell_direction, cell_magnitude, hist_bins):
        HOG_cell_hist = np.zeros(shape=(hist_bins.size))
        cell_size = cell_direction.shape[0]
        
        for row_idx in range(cell_size):
            for col_idx in range(cell_size):
                curr_direction = cell_direction[row_idx, col_idx]
                curr_magnitude = cell_magnitude[row_idx, col_idx]
        
                diff = np.abs(curr_direction - hist_bins)
                
                if curr_direction < hist_bins[0]:
                    first_bin_idx = 0
                    second_bin_idx = hist_bins.size-1
                elif curr_direction > hist_bins[-1]:
                    first_bin_idx = hist_bins.size-1
                    second_bin_idx = 0
                else:
                    first_bin_idx = np.where(diff == np.min(diff))[0][0]
                    temp = hist_bins[[(first_bin_idx-1)%hist_bins.size, (first_bin_idx+1)%hist_bins.size]]
                    temp2 = np.abs(curr_direction - temp)
                    res = np.where(temp2 == np.min(temp2))[0][0]
                    if res == 0 and first_bin_idx != 0:
                        second_bin_idx = first_bin_idx-1
                    else:
                        second_bin_idx = first_bin_idx+1
                
                first_bin_value = hist_bins[first_bin_idx]
                second_bin_value = hist_bins[second_bin_idx]
                HOG_cell_hist[first_bin_idx] = HOG_cell_hist[first_bin_idx] + ( np.abs(curr_direction - first_bin_value)/(180.0/hist_bins.size) ) * curr_magnitude
                HOG_cell_hist[second_bin_idx] = HOG_cell_hist[second_bin_idx] + ( np.abs(curr_direction - second_bin_value)/(180.0/hist_bins.size) ) * curr_magnitude
        return HOG_cell_hist

    @staticmethod
    def hog(magnitude,direction):
        #for magnitude.
        cell_direction = direction[222:230, 139:147]
        cell_magnitude = magnitude[222:230, 139:147]
        HOG_cell_hist = OpImage.HOG_cell_histogram(cell_direction, cell_magnitude, np.array([10,30,50,70,90,110,130,150,170]))

        plt.bar(x=np.arange(9), height=HOG_cell_hist, align="center", width=0.8)
        plt.show()