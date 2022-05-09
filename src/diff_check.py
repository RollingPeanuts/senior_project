import cv2
import time
import numpy as np

# load images
cap = cv2.VideoCapture("/dev/video2")
ret, img = cap.read()

i = 0
while(i <= 500):
    ret, img2 = cap.read()
    i += 1

#img = cv2.imread("image.png")
#img2 = cv2.imread("image2.png")
# create copy for image comparison
img2_ori = img2.copy()
# subtract to get difference
diff =  cv2.subtract(img, img2)
# create grayscale of diff
gray =  cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
# create a mask for the non black values (above 10) 
ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
# find contours in mask
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# calculate the center of each contour using the boundingrect
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    centerX = x+ int(w/2)
    centerY = y+ int(h/2)
    print(centerX)
    print(centerY)

    # draw blue dot in center
    cv2.circle(img2,(centerX, centerY),5,(255,0,0),-1)

#show images
cv2.imshow("img", img)
cv2.imshow("img2", img2_ori)
cv2.imshow("diff", thresh1)
cv2.imshow("result", img2)

cv2.waitKey(0)
cv2.destroyAllWindows() 
