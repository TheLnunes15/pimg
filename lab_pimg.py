########################################
#
# Nome: Lucas dos Santos Nunes
# Matricula: 202011000755
# E­mail: lucas.nunes@dcomp.ufs.br
#
########################################

import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt


# Q.2
def imread(n_arq):
    return mp.imread(n_arq,np.uint8)


# Q.7
def imshow(img):    
    if(nchannels(img) == 1):
        plt.imshow(img, vmin=0, vmax=255, cmap='gray', interpolation='nearest')        
    else:
        plt.imshow(img.astype('uint8'), interpolation='nearest')        
    plt.show()


# Q.3
def nchannels(img):
    if len(img.shape) == 2: 
        return 1
    return img.shape[2]


# Q.4
def size(img): 
    tam = [len(img[0]), len(img)]
    return tam


# Q.5
def rgb2gray(img):
    if nchannels(img) == 1:
        return img
    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]   
    return np.uint8((0.299 * r) + (0.587 * g) + (0.114 * b) / (0.299+0.587+0.114))
    

# Q.6
def imreadgray(n_arq):
    img = imread(n_arq)
    if nchannels(img) == 1:
        return img
    
    elif nchannels(img) >= 3:
       return rgb2gray(img) 


# Q.8
def thresh(img, limiar):
    imgOut = []
    if nchannels(img) == 1:
        for col in img:
            imgOut.append([255 if p >= limiar else 0 for p in col])
    else:
        for col in img:
            line = []
            for p in col:
                if p[0] >= limiar[0]:
                    a0 = 255
                else:
                    a0 = 0
                if p[1] >= limiar[1]:
                    a1 = 255
                else:
                    a1 = 0
                if p[2] >= limiar[2]:
                    a2 = 255
                else:
                    a2 = 0
                line.append([a0, a1, a2,])
            imgOut.append(line)
    return np.asarray(imgOut, np.uint8)


# Q.9
def negative(img):
    return 255 - img;


# Q.10
def contrast(img, r, m):
	out = r * (img - m) + m
	out = [[255 if x > 255 else x for x in arr] for arr in out]
	out = [[0 if x < 0 else x for x in arr] for arr in out]

	return np.asarray(out, np.uint8)


# Q.11
def hist(img):  
    N, M = size(img) 
    if nchannels(img) > 1: 
        r = np.zeros(256)
        g = np.zeros(256)
        b = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                r[img[i,j, 0]] += 1 
                g[img[i,j, 1]] += 1
                b[img[i,j, 2]] += 1
        return np.array([r, g, b])
    else:
        c = np.zeros(256)
        for i in np.arange(N):
            for j in np.arange(M):
                c[img[i,j]] += 1
        return np.array([c])


# Q.12 e Q.13
def showhist(histo, bin=1):
    ''' mostra um gráfico de barras para o histograma da imagem '''
    width = 0.25
    n_bin = 256//bin
    plt.figure(figsize=(20,10))
    plt.xlabel('pixels')
    plt.ylabel('count')
    plt.title('Histograma')
    print(len(histo))
    if len(histo) > 1:
        newHisto = np.zeros((3, n_bin+1))
        for i in np.arange(0, 256, bin):
            newHisto[0, i//bin] = sum(histo[0, i:i+bin])
            newHisto[1, i//bin] = sum(histo[1, i:i+bin])
            newHisto[2, i//bin] = sum(histo[2, i:i+bin])
        b1 = np.arange(n_bin+1)
        b2 = [x + width for x in b1]
        b3 = [x + width for x in b2] 
        plt.bar(b1, newHisto[0], width, alpha= 0.7, color='red', align='center')
        plt.bar(b2, newHisto[1], width, alpha= 0.7, color='green', align='center')
        plt.bar(b3, newHisto[2], width, alpha= 0.7, color='blue', align='center')
        plt.show()
    
    else:
        newHisto = np.zeros((1, n_bin+1))
        for i in np.arange(0, 256, bin):
            newHisto[0, i//bin] = sum(histo[0, i:i+bin])
        b1 = np.arange(n_bin+1)
        plt.bar(b1, newHisto[0], width, alpha= 0.7, color='gray', align='center')
        plt.show()


# Q.14
def histeq(img):
    histo = hist(img)
    fdp = np.cumsum(histo) / np.sum(histo)
    e = (fdp[img] * 255).astype('uint8')   
    return e


# Q.15
def convolve(img, mask):
	Sx, Sy = size(img)
	
	a = len(mask)
	b = len(mask[0])
	a2 = int(a/2)
	b2 = int(b/2)
	outAux = np.zeros((Sy, Sx), np.int32)
	
	for i in range(Sy):
		for j in range(Sx):
			g = 0.0
			for s in range(a):
				for t in range(b):
					x = j+t-a2
					y = i+s-b2
					x = min(max(x, 0), Sx-1)
					y = min(max(y, 0), Sy-1)
					g += (mask[s][t] * img[y][x])
			outAux[i][j] = abs(int(g))
	
	out = [[255 if x > 255 else x for x in arr] for arr in outAux]
	out = [[0 if x < 0 else x for x in arr] for arr in out]
	
	return np.asarray(out, np.uint8)


# Q.16
def maskBlur():
  	return np.dot([[1,2,1], [2,4,2], [1,2,1]], 1/16)


# Q.17
def blur(img):
  	blur_img = convolve(img, maskBlur())
  	return blur_img


# Q.18
def seSquare3():
    return np.ones((3,3), np.uint8)


# Q.19
def seCross3():
    return np.array([[0,1,0], [1,1,1], [0,1,0]], np.uint8)


# Q.20
def erode(image,eleBin):    
    n = len(eleBin)
    m = len(eleBin[0])
   
    menorimg = np.copy(image)    

    a = ( n-1 ) // 2
    b = ( m-1 ) // 2

    if(nchannels(image)==1):
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = []
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):

                                menor.append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)])
                
                menorimg[i][j] = min(menor)
   
    if(nchannels(image)==3):        
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = [[],[],[]]
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):
                                menor[0].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][0])                                
                                menor[1].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][1])                                
                                menor[2].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][2])

                menorimg[i][j][0] = min(menor[0])
                menorimg[i][j][1] = min(menor[1])
                menorimg[i][j][2] = min(menor[2])

    return menorimg


#Q.21
def dilate(image,eleBin):    
    n = len(eleBin)
    m = len(eleBin[0])
   
    menorimg = np.copy(image)    

    a = ( n-1 ) // 2
    b = ( m-1 ) // 2
    
    if(nchannels(image)==1):
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = []
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):
                                menor.append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)])
                
                menorimg[i][j] = max(menor)
   
    if(nchannels(image)==3):        
        for i in range(len(image)):
            for j in range (len(image[i])):
                menor = [[],[],[]]
                for k in range(-a, a+1): 
                        for l in range (-b,b+1):
                            
                            if( eleBin[k+1][l+1] != 0):
                                menor[0].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][0])
                                menor[1].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][1])            
                                menor[2].append( image[ min(max(i+k, 0), image.shape[0]-1)][min(max(j+l, 0), image.shape[1]-1)][2])

                menorimg[i][j][0] = max(menor[0])
                menorimg[i][j][1] = max(menor[1])
                menorimg[i][j][2] = max(menor[2])

    return menorimg


# imread('lena1.jpg')
# imshow(imread('lena1.jpg'))
# size(imread('lena1.jpg'))
# imreadgray(imread('lena1.jpg'))
# nchannels(imread('lena1.jpg'))
# rgb2gray(imread('lena1.jpg'))

# Q.8
# imshow(thresh(imreadgray('img.png'),90))
# imshow(thresh(imreadgray('lena1.jpg'),90))

# Q.9
# imshow(negative(imreadgray('img.png')))
# imshow(negative(imreadgray('lena1.jpg')))

# Q.10
# imshow(contrast(imreadgray('lena1.jpg'),4,5))

# Q.12 e Q.13
# showhist(hist((imreadgray('lena1.jpg'))))
# showhist(hist(imreadgray('lena1.jpg')),4)

# Q.15
# imshow(convolve(imreadgray('lenas.jpg'),maskBlur()))
# imshow(convolve(imreadgray('lena1.jpg'),maskBlur()))
# imshow(convolve(imreadgray('img.png'),maskBlur()))

# Q.20 e Q.21
# imshow(erode(imread('lena1.jpg'),seCross3()))
# imshow(erode(imreadgray('lena1.jpg'),[[0,1,0], [1,1,1], [0,0,0]]))
# imshow(dilate(imread('lena1.jpg'),seCross3()))
# imshow(dilate(imreadgray('lena1.jpg'),[[0,1,0], [1,1,1], [0,0,0]]))
