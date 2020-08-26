########################################
#
# Nome: Lucas dos Santos Nunes
# Matricula: 202011000755
# E­mail: lucas.nunes@dcomp.ufs.br
#
########################################

from math import *
import numpy as np
import matplotlib.image as mp
import matplotlib.pyplot as plt


def imread(n_arq):
    return mp.imread(n_arq,np.uint8)


def imshow(img):    
    if(nchannels(img) == 1):
        plt.imshow(img, vmin=0, vmax=255, cmap='gray', interpolation='nearest')        
    else:
        plt.imshow(img.astype('uint8'), interpolation='nearest')        
    plt.show()


def imshow2(img):    
    if(nchannels(img) == 1):
        plt.imshow(img, cmap='gray', interpolation='nearest')        
    else:
        plt.imshow(img.astype('uint8'), interpolation='nearest')        
    plt.show()


def nchannels(img):
    if len(img.shape) == 2:
        return 1
    else: return img.shape[2]


def rgb2gray(img):
    if nchannels(img) == 1:
        return img    
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]   
    return np.uint8((0.299 * r) + (0.587 * g) + (0.114 * b) / (0.299+0.587+0.114))
    

#-------------------------------------------------------------------------------------------------------------------------------------------

# Q.1
# Transformacao Direta Discreta Bidimensional de Fourier

def dft2(x):
    N2 = len(x)
    N1 = len(x[0])
    X = np.zeros((N2,N1),dtype=complex)
    for u in range(N1):
        for v in range(N2):
            for n1 in range(N1):
                comp = complex()
                for n2 in range(N2):
                    comp += x[n2][n1] * complex(cos(2 * pi * v * n2 / N2), -sin(2 * pi * v * n2 / N2))
                X[v][u] += comp * complex(cos(2 * pi * u * n1 / N1), -sin(2 * pi * u * n1 / N1))
    X /= N1*N2
    return X


def idft2(X):
    N2 = len(X)
    N1 = len(X[0])
    x = np.zeros((N2,N1),dtype=complex)
    for n1 in range(N1):
        for n2 in range(N2):
            for u in range(N1):
                comp = complex()
                for v in range(N2):
                    comp += X[v][u] * complex(cos(2 * pi * v * n2 / N2), sin(2 * pi * v * n2 / N2))
                x[n2][n1] += comp * complex(cos(2 * pi * u * n1 / N1), sin(2 * pi * u * n1 / N1))
    return x


#im = imread("lenas.jpg")
#im = imread("50x50.gif")
#gray = rgb2gray(im)
#df = dft2(gray)
#idf = idft2(df)
#imshow(df.real)
#imshow(idf.real)
#imshow2(df.real)
#imshow2(idf.real)
#imshow(df.imag)
#imshow(idf.imag)
#imshow2(df.imag)
#imshow2(idf.imag)

#npfft = np.fft.fft2(gray)
#real = npfft.real
#npffti = np.fft.ifft2(real)
#reali = npffti.real
#imshow(real)
#imshow(reali)
#imshow2(real)
#imshow2(reali)

#-------------------------------------------------------------------------------------------------------------------------------------------

# Q.4
# Transformacao Rapida Unidimensional de Fourier

"""

def ordenar(x):
    num = len(x)
    j = num // 2
    for i in range(1, num - 2):
        if (i < j):
            x[i], x[j] = x[j], x[i]
        k = num / 2
        while (j >= k):
            j -= k
            k /= 2
        j = int(j + k)


def fft(x):
    ordenar(x)
    N = len(x)
    M = int(log(N,2))

    x = np.zeros((N,M), dtype=complex)
    
    for base in range(M):
        qtds = 1 << base
        i = 0
        for group in range(0, N, qtds << 1):
            for qtd in range(qtds):
                k = x[i+(1<<base)] * complex(cos(2 * pi * qtd / (1 << (base+1))), -sin(2 * pi * qtd / (1 << (base+1))))
                x[i + (1 << base)] = x[i] - k
                x[i] += k
                i += 1
            i += qtds
    return x


def ifft(X):
    ordenar(X)
    N = len(X)
    M = int(log(N,2))

    X = np.zeros((N,M), dtype=complex)
    
    for base in range(M):
        qtds = 1 << base
        i = 0
        for group in range(0, N, qtds << 1):
            for qtd in range(qtds):
                k = X[i+(1<<base)] * complex(cos(2 * pi * qtd / (1 << (base+1))), sin(2 * pi * qtd / (1 << (base+1))))
                X[i + (1 << base)] = X[i] - k
                X[i] += k
                i += 1
            i += qtds
    X /= N


def fft2(x):
    N2 = len(x)
    N1 = len(x[0])
    
    for v in range(N2):
        fft(x[v])
    for u in range(N1):
        fft(x[:, u])
    return x


def ifft2(X):
    N2 = len(X)
    N1 = len(X[0])
    
    for n2 in range(N2):
        ifft(X[n2])
    for n1 in range(N1):
        ifft(X[:,n1])
    return X

"""

def dft(x):
    N = len(x)
    X = np.array([complex() for i in range(N)])
    for k in range(N):
        for n in range(N):
            X[k] += x[n]*complex(cos(2 * pi * k * n / N), -sin(2 * pi * k * n / N))
    X /= N
    return X


def eh_potencia_de_2(num):
	return num and not(num & (num-1))


def fft( arr ):
	arr = np.asarray( arr, dtype=np.complex )
	N = arr.shape[0]

	if(not eh_potencia_de_2(N)):
		print( arr.shape )
		raise ValueError( "Eh necessario que N seja potencia de 2" )
	
	if( N <= 16 ):
		return dft( arr )
	else:
		even_mid = fft( arr[::2] )
		odd_mid = fft( arr[1::2] ) 
		comp = np.exp( -2j * np.pi * np.arange( N ) / N )

		fu = even_mid + odd_mid * comp[ :int(N//2) ]
		fu_k = even_mid + odd_mid * comp[ int(N//2): ]

		return np.concatenate( [ fu, fu_k ] )


def ifft( arr_fourrier ):
	arr = np.asarray( arr_fourrier, dtype=np.complex )
	arr = np.conjugate( arr_fourrier )
	arr = fft( arr )
	arr = np.conjugate( arr )
	arr = arr / arr_fourrier.shape[0]
	return arr


def fft2( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = fft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = fft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = fft2( img[:, :, canal] )
			return img_fourrier


def ifft2( img ):
	if( isinstance( img, np.ndarray ) ):
		if( nchannels( img ) == 1 ):
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for i in range( img.shape[0] ):
				img_fourrier[i, :] = ifft( img[i, :] )
			
			for j in range( img.shape[1] ):
				img_fourrier[:, j] = ifft( img_fourrier[:, j] )
			
			return img_fourrier
		elif( nchannels( img ) > 1 ):
			img_fourrier = np.zeros( img.shape, dtype=np.complex )
			for canal in range( 3 ):
				img_fourrier[:, :, canal] = ifft2( img[:, :, canal] )
			return img_fourrier



img = imread("lena1.jpg")
#img = imread("sin8d.gif")
#img = imread("50x50.gif")
imggray = rgb2gray(img)

teste = fft2(imggray)
tt = ifft2(teste)
imshow2(teste.real)
imshow2(tt.real)

npfft = np.fft.fft2(imggray)
real = npfft.real
npffti = np.fft.ifft2(real)
reali = npffti.real
imshow(real)
imshow(reali)
imshow2(real)
imshow2(reali)
