import cv2
import numpy as np
from time import sleep

def thresholdBetweenValues(img, thresh_min, thresh_max):
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(img, thresh_min, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(img, thresh_max, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(threshold, threshold2)


def thresholdInRange(img, threshold_range):
    return thresholdBetweenValues(img, threshold_range[0], threshold_range[1])

def findBackground(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blue = [0.5 * 180, 0.65 * 180]
    background = thresholdInRange(h, blue)
    background = cv2.morphologyEx(background, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8))
    contours, hierarchy = cv2.findContours(background, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # Find two biggest contours
    max_area_index = 0
    second_max_area_index = 0
    max_area = 0
    second_max_area = 0
    for i, cont in enumerate(contours):
        tmp_area = cv2.contourArea(cont)
        if tmp_area > max_area:
            second_max_area = max_area
            second_max_area_index = max_area_index
            max_area = tmp_area
            max_area_index = i
        elif tmp_area > second_max_area:
            second_max_area = tmp_area
            second_max_area_index = i

    # Największym konturem jest prawie zawsze cała plansza.
    # Drugim co do wielkości jest plansza z wyciętą wodą, która by nas bardziej interesowała.
    # Jednak na niektórych zdjęciach oba te kontury się zlewają w jeden, więc nie możemy zawsze brać tego mniejszego.
    # If the second biggest contour is inside the biggest one take the inside one
    if hierarchy[0][second_max_area_index][3] == max_area_index:
        best_contour = contours[second_max_area_index]
    else:
        best_contour = contours[max_area_index]
    return best_contour

def drawContourOnimg(img, contour):
    cv2.drawContours(img, [contour], -1, 255, cv2.FILLED)
    return img

def cutBackground(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def getBoard(img):
    contour = findBackground(img)
    img_size = img.shape[:2]
    mask = np.zeros(img_size, dtype=np.uint8)
    contour_hull = cv2.convexHull(contour, False)
    # If the contour is not solid draw the covex hull instead
    if cv2.contourArea(contour) > 0.5 * cv2.contourArea(contour_hull):
        mask = drawContourOnimg(mask, contour)
    else:
        mask = drawContourOnimg(mask, contour_hull)
    return cutBackground(img, mask), mask

def getCorners(edges, orig_img):
    gray = edges
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = np.float32(edges)
    # corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    # corners = np.int0(corners)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(orig_img,(x,y),3,255,-1)
    dst = cv2.cornerHarris(gray,21,21,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None,iterations=2)
    # Threshold for an optimal value, it may vary depending on the img.
    orig_img = cv2.subtract(orig_img, edges.astype)
    orig_img[dst>0.075*dst.max()]=[0,0,255]
    return orig_img

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged

def getLines(img, orig_img):
    gray = img
    # kernel_size = 3
    # blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    # return blur_gray
    low_threshold = 20
    high_threshold = 550
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # edges = auto_canny(gray)
    cv2.imshow("canny", edges)
    return edges

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    linesP = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    for i in range(0, len(linesP)):
            lin = linesP[i][0]
            cv2.line(orig_img, (lin[0], lin[1]), (lin[2], lin[3]), (0,0,255), 3, cv2.LINE_AA)
    # for line in lines:
    #     for x1,y1,x2,y2 in line:
    #         cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    # lines_edges = cv2.addWeighted(img, 0, line_image, 1, 0)
    return orig_img

def watershed(gray_img):
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.2, 100)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(gray_img, (x, y), r, (0, 255, 0), 4)
    # gray_img = cv2.dilate(gray_img, kernel)
    # kernel1 = np.ones((3,3), dtype=np.uint8)
    # gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_GRADIENT, kernel1)
    # kernel1 = np.ones((3,3), dtype=np.uint8)
    # gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel1)
    
    return gray_img

def getMarkers(img):
    h, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blue = [0.5 * 180, 0.65 * 180]
    background = thresholdInRange(h, blue)
    return background

def main():
    vid = cv2.VideoCapture(0)
    sleep(1)
    _, img = vid.read()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('img', img)
    board, background = getBoard(img)
    markers = getMarkers(board)
    cv2.imshow('board', markers)
    img_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    
    # kernel_size = 5
    # img_gray = cv2.GaussianBlur(img_gray,(kernel_size, kernel_size),0)
    # bilateral = cv2.bilateralFilter(img_gray, 5, 75, 75)

    ret, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, background)

    kernel = np.ones((3,3), dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

    # thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            # cv2.THRESH_BINARY_INV,11,5)
    cv2.imshow("thresh", thresh)

    img = getLines(img_gray, board)
    cv2.imshow("lines", img)
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    dist_transform = cv2.distanceTransform(img,cv2.DIST_L2,5)
    cv2.imshow('dt', dist_transform)
    markers = np.uint8(markers)
    unknown = cv2.subtract(background,markers)
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[unknown==255] = 0

    img = cv2.watershed(color_img, markers)
    # img[markers==-1] = [255,0,0]
    img = img.astype(np.uint8)
    cv2.imshow('ws', img)
    img = getCorners(img, board)
    
    cv2.imshow("corners", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

    