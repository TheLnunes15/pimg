import lab_pimg as w

def testeConvolve(file):
    img = w.imreadgray(file)
    k = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    ans = w.convolve(img, k)
    w.imshow(ans)
    return ans
    
def testeBlur(file):
	img = w.imreadgray(file)
	w.imshow(img)
	ans = w.blur(img)
	w.imshow(ans)

######### testes ############
# Q1 imread()
img1 = w.imread('img.png')
print (type(img1))

# Q7 imshow()
w.imshow(img1)

# Q3 pegando numero de canais
nCh = w.nchannels(img1)
print (nCh)

# Q4 
print (w.size(img1))

# Q5
w.rgb2gray(img1)

# Q6
img1Gray = w.imreadgray('img.png')

# Q8 threshold
imgT = w.thresh(img1Gray, 200)
w.imshow(imgT)
w.imshow(img1Gray)
#imgT = thresh(img1, [100, 250, 250])
#imshow(imgT)

# Q9 
imgN = w.negative(img1Gray)
w.imshow(imgN)
w.imshow(img1Gray)

# Q10
imgC = w.contrast(img1Gray, 4, 20)
w.imshow(imgC)
w.imshow(img1Gray)

# Q11
h = w.hist(img1Gray)

# Q12 e Q13
w.showhist(h)
w.showhist(h, 3)

# Q14
imgH = w.histeq(img1Gray)
w.imshow(imgH)
w.imshow(img1Gray)

imgB = w.blur(img1Gray)
w.imshow(imgH)
w.imshow(img1Gray)

out1 = w.erode(img1Gray, w.seSquare3())
out2 = w.erode(img1Gray, w.seCross3())

w.imshow(img1Gray)
w.imshow(out1)
w.imshow(out2)

out1 = w.dilate(img1Gray, w.seSquare3())
out2 = w.dilate(img1Gray, w.seCross3())

w.imshow(img1Gray)
w.imshow(out1)
w.imshow(out2)
