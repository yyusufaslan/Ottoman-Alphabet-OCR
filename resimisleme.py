import cv2 as cv
import numpy as np

pencere="pencere"
cv.namedWindow(pencere)

def nothing(x):
    pass

asin = 2
genisle = 3
kernelSize = 2
cv.createTrackbar("asindir", pencere, 0, 20, nothing)
cv.createTrackbar("genislet", pencere, 0, 20, nothing)
cv.createTrackbar("kernel", pencere, 0, 10, nothing)
cv.setTrackbarPos("asindir", pencere, asin)
cv.setTrackbarPos("genislet", pencere, genisle)
cv.setTrackbarPos("kernel", pencere, kernelSize)

def main():
	while(1):
		orig = cv.imread("göt.jpeg")
		#gray = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
		ret, img = cv.threshold(gray,0,255, cv.THRESH_TOZERO|cv.THRESH_OTSU)
		kernel = np.ones((2*cv.getTrackbarPos("kernel", pencere)+1,2*cv.getTrackbarPos("kernel", pencere)+1),np.uint8)
		img = cv.erode(img,kernel,iterations = cv.getTrackbarPos("genislet", pencere))
		img = cv.dilate(img,kernel,iterations = cv.getTrackbarPos("asindir", pencere))


		cv.imshow(pencere,img)
		k = cv.waitKey(1) & 0xFF
		if k == 'q':
			break


main()
