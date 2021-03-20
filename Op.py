import cv2
import numpy as np
from math import sqrt,atan2,ceil,pi
import matplotlib.pyplot as plt
from skimage import draw

def mod(n):
    if(n>0):
        return n
    else:
        return -1*n
class OpImage:
    
    
                
    @staticmethod
    def janela(image,x,y,n):
        print("Janela",n,"x",n,"em",x,":",y)
        """
        IMPLEMENTAÇÃO DA JANELA
        TAMBÉM É VISTO COMO REGION-OF-INTEREST
        """
        #if n % 2 == 0: raise Exception("'N' deve ser Ímpar")
        ret = []
        lm = int(n-1-(n-1)/2)
        for j in range( -lm,lm+1 ):
            ret.append([])
            for i in range( -lm,lm+1 ):
                ret[-1].append(image[y+j][x+i])
        return ret
        

    @staticmethod
    def threshold(image,threshold,inverse=False):
        """
        Percorre a imagem(grayscale), consultando o valor de cada pixel, e atribuindo
        255 se o valor for maior que o 'threshold', senão atribui 0
        """
        out = image.copy()
        match = 255 if inverse == False else 0
        no_match = 0 if inverse == False else 255
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if mod(image[i][j]) < threshold:
                    out[i][j] = no_match
                '''if image[i][j] >= threshold:
                    out[i][j] = match
                else:
                    out[i][j] = no_match'''
        return out

    @staticmethod
    def rescale(image,min,max):
        """
        remapea os valores de uma lista para entre os parametros 'min' e 'max'
        """
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
    def gradienteMedia(grad1):
        """
        Código para calcular média
        pode ser utilizado para qualquer matriz
        É utilizado pelo algoritmo de threshold baseado em médias
        """
        med = 0
        ci = 1
        for x in range(grad1.shape[0]):
            for y in range(grad1.shape[1]):
                if grad1[x][y] > 0:
                    med+=grad1[x][y]
                    ci+=1
        med/=ci
        return med

    @staticmethod
    def mag_direction(im1,im2):
        """
        calcula a transformação de coordenadas cartesianas
        para polares, em formato de magnitude ( 'r' )
        e direcao ( theta )
        """
        print("Magnitude e direção")
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
    def threshold_media(image,k):
        """
        Algoritmo de threshold utilizado para remoção de elementos do gradiente.
        Este algoritmo calcula a limiarização para uma lista de coeficientes,
        utilizando como limiar a média da magnitude subtraida do desvio padrão 
        da matriz de gradientes ( desvio multiplicado a uma constante )
        """
        print("Threshold por  média das magnitudes")
        sigma = np.std(image)
        #out=[np.zeros((image.shape[0],image.shape[1]))]*len(k)
        out=np.ndarray(( len(k),image.shape[0],image.shape[1] ))
        rn = range(len(k))
        gradMedia = OpImage.gradienteMedia(image)
        print(gradMedia)
        T = [ np.max([gradMedia - (kk*sigma),0.1]) for kk in k ]
        print(T)
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                for ti in rn:
                    if image[x][y] >= T[ti]:
                        out[ti][x][y]=255
                    else:
                        out[ti][x][y]=0
        return out


    #https://www.pyimagesearch.com/2016/07/25/convolutions-with-opencv-and-python/
    #https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.rescale_intensity
    @staticmethod
    def convolucao(image, kernel):
        """
        Algoritmo principal de "janela deslizante"
        Essa função percorre todos os pixels da imagem (desconsiderando bordas)
        calculando a regiao de interesse (ROI/janela) multiplicada aos coeficientes
        do kernel ( máscara / núcleo )
        """
        print("convolucao do filtro ",kernel)
        iW, iH = image.shape[0],image.shape[1]
        kW, kH = kernel.shape[0],kernel.shape[1]
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
        """
        Algoritmo cria uma matriz identidade, divide pelo quadrado do
        comprimento do lado da matriz, e utiliza a convolução para 
        percorrer a imagem aplicando-a como kernel
        """
        print("média por convolução de matriz identidade")
        media_matriz = np.ones((n,n)) * (1 / n**2)
        return OpImage.convolucao(image,media_matriz)

    @staticmethod
    def media(image,n): # pegando os cantos
        """
        Algoritmo para calcular a média de regiões da imagem com um 
        tamanho N x N, pegando os cantos da imagem
        """
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
    def media_nomargin(image,n): # não pega os cantos
        """
        mesmo principio do algoritmo de OpImage.media, porém não considera os cantos
        (começa a contar apartir de uma margem ( margem = (n-1)/2, onde n = tamanho do kernel ) )
        """
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
        """
        Seleciona pixels, dada a diferença entre a média da região NxN sendo maior que o range
        """
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
    def rgbSum(image,hog):
        """
        Substitui um canal de cor da imagem, pelo parâmetro da matriz 'hog'
        """
        out=image.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                #out[x][y][0] = hog[x][y]
                out[x][y][1] = hog[x][y]
        return out
        
    @staticmethod
    def sobelRgb(sobelx,sobely):
        """
        Soma as matrizes sobelx e sobely
        em canais de cores
        """
        out=np.zeros((sobelx.shape[0],sobelx.shape[1],3),np.uint8)
        for x in range(sobelx.shape[0]):
            for y in range(sobelx.shape[1]):
                #out[x][y][0] = hog[x][y]
                out[x][y][2] = sobelx[x][y]
                out[x][y][1] = sobely[x][y]
                out[x][y][0] = 0
        return out


    @staticmethod
    def merge_min(image,subtr):
        """
        Função para subtrair matrizes (X,Y,1) sem normalizar
        """
        nshape = image.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                nshape[x][y]=np.max(image[x][y]-subtr[x][y],0)
        return nshape

    @staticmethod
    def merge_max(image,add):
        """
        Função para somar matrizes (X,Y,1) sem normalizar
        """
        nshape = image.copy()
        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                nshape[x][y]=np.min([int(image[x][y])+int(add[x][y]) ,255])
        return nshape

    


    @staticmethod
    def grayscale(image):
        """
        Cria grayscale da imagem
        """
        print("Grayscale da imagem")
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
        """
        Algoritmo para calcular histograma de uma região específica 
        da matriz de direções e magnitudes
        código original: https://github.com/ahmedfgad/HOGNumPy/blob/master/HOG.py

        """
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
    def normalizeHog(cell_list):
        """
        Normaliza os valores de uma lista
        Utilizado para normalizar os valores de magnitudes do HOG
        """
        sum2 = 0.0
        for i in cell_list:
            sum2+=i**2
        sum2=sqrt(sum2)
        if sum2 >= 0:
            sum2+=1
        cell_list /= sum2
        return cell_list

    @staticmethod
    def drawHOG(hist, csx=8, csy=8, signed_orientation=False):
        """
        Desenhando o histograma no tamanho da imagem original, 
        em células 8x8 os vetores normalizados do histograma
        ( não foi feito interpolação tripla, operação necessária para obter
        resultados eliminando o 'ruído' )
        """
        print("desenhando hog")
        if signed_orientation:
            max_angle = 2*np.pi
        else:
            max_angle = np.pi
        
        n_cells_y, n_cells_x, nbins = hist.shape
        sx, sy = n_cells_x*csx, n_cells_y*csy
        #center = csx//2, csy//2
        #b_step = max_angle / nbins

        radius = min(csx, csy) // 2 - 1
        hog_image = np.zeros((sx, sy), dtype=float)
        for x in range(n_cells_x):
            for y in range(n_cells_y):
                for o in range(nbins):
                    centre = tuple([y * csy + csy // 2, x * csx + csx // 2])
                    dx = radius * np.cos(o*nbins)
                    dy = radius * np.sin(o*nbins)
                    rr, cc = draw.line(int(centre[0] - dy),
                                    int(centre[1] - dx),
                                    int(centre[0] + dy),
                                    int(centre[1] + dx))
                    hog_image[cc, rr] += hist[y, x, o]
        return hog_image

                

    @staticmethod
    def hog(magnitude,direction):
        """
        Calculando o histograma da imagem inteira, em várias regiões de 8x8
        """
        print("Calculando hog da imagem")
        #angles=np.array([10,30,50,70,90,110,130,150,170])
        #angles=np.array([0,20,40,60,80,100,120,140,160])
        angles=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])
        angrange = range(len(angles))
        lmy,lmx = magnitude.shape[:2]
        hog_cells = np.ndarray( ((lmx//8),(lmy//8),len(angles)) )
        ii,jj=0,0
        for i in range(0,lmx-1,8):
            jj=0
            for j in range(0,lmy-1,8):
                '''cell_direction = direction[178:186, 138:146]
                cell_magnitude = magnitude[178:186, 138:146]'''
                cell_direction = direction[j:j+7, i:i+7]
                cell_magnitude = magnitude[j:j+7, i:i+7]
                HOG_list = OpImage.HOG_cell_histogram(cell_direction, cell_magnitude, angles)
                norm_hog = OpImage.normalizeHog( HOG_list )
                for nh in angrange:
                    hog_cells[ii][jj][nh]=norm_hog[nh]
                jj+=1
            ii+=1
        return hog_cells

    @staticmethod
    def hog_region(magnitude,direction,x0,y0,x1,y1):
        """
        Calcula o hog em uma região especificada
        ao invés de calcular a imagem inteira
        """
        angles=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])
        cell_direction = direction[x0:x0+7, y0:y0+7]
        cell_magnitude = magnitude[x0:x0+7, y0:y0+7]
        HOG_list = OpImage.HOG_cell_histogram(cell_direction, cell_magnitude, angles)
        norm_hog = OpImage.normalizeHog( HOG_list )
        return norm_hog

    @staticmethod
    def agrupar_hog(hog_data,x=0,y=0,block_size=4):
        """
        Agrupa as magnitudes de grupos de células (block_size x block_size)
        e retorna (normalizado) o novo vetor somatório de magnitudes de todas as 
        regiões
        """
        hog_angles=hog_data.shape[2]
        hog_vector = np.zeros((hog_angles))
        for i in range(x,x+block_size):
            for j in range(x,x+block_size):
                for k in range(hog_angles):
                    hog_vector[k] = hog_data[x][y][k]
        hog_vector = OpImage.normalizeHog(hog_vector)
        return hog_vector


    @staticmethod
    def drawHOGHistogram(hog_hist):
        """
        Desenhar gráfico em barras das magnitudes dos dados do histograma
        """
        x = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170])
        #x = np.arange(len(hog_hist))
        fig, axes = plt.subplots(ncols=1, nrows=1)
        plt.title('HOG')
        plt.xlabel('Angulos')
        plt.ylabel('Magnitudes')
        axes.bar(x, hog_hist,width=8,align="center")
        axes.set_xticks(x)
        #axes.set_xticklabels(['Positive'])
        plt.legend()
        plt.show()