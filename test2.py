import numpy as np
import cv2

image = cv2.imread("C:\\Users\\PYN\\Desktop\\xueweilunwen\\MyProject\\output\\desk\\data\\00001.jpg")#读入图像
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
# cv2.imshow("Image",gray)#显示图像
# cv2.waitKey()
#
# #Canny边缘检测
# canny = cv2.Canny(gray,30,150)
# cv2.imshow("Canny",canny)
# cv2.waitKey()

spatialRad = 50
colorRad = 50
maxPryLevel = 1

def fill_color_demo(image):
    copyIma = image.copy()
    h, w = image.shape[:2]
    print(h, w)
    mask = np.zeros([h+2, w+2], np.uint8)
    cv2.floodFill(copyIma, mask, (30, 30), (0, 255, 255), (100, 100, 100), (50, 50, 50), cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow("fill_color", copyIma)

dst = cv2.pyrMeanShiftFiltering(image, spatialRad, colorRad, maxLevel=maxPryLevel)
cv2.imshow("dst", dst)
fill_color_demo(dst)
cv2.waitKey(0)