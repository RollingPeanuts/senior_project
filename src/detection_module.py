import cv2
import numpy as np
import mediapipe as mp
import time
from math import dist
from google.cloud import pubsub_v1

# Source for get_board_and_background and helper functions:
# https://github.com/Vieja/Catan-Image-Recognition/blob/master/catan.py
def threshold_between_values(img, thresh_min, thresh_max):
    # Finding two thresholds and then finding the common part
    _, threshold = cv2.threshold(img, thresh_min, 255, cv2.THRESH_BINARY)
    _, threshold2 = cv2.threshold(img, thresh_max, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(threshold, threshold2)

def threshold_in_range(img, threshold_range):
    return threshold_between_values(img, threshold_range[0], threshold_range[1])

def find_background(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blue = [0.5 * 180, 0.65 * 180]
    background = threshold_in_range(h, blue)
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
    # If the second biggest contour is inside the biggest one take the inside one
    if hierarchy[0][second_max_area_index][3] == max_area_index:
        best_contour = contours[second_max_area_index]
    else:
        best_contour = contours[max_area_index]
    return best_contour

def draw_contours_on_img(img, contour):
    cv2.drawContours(img, [contour], -1, 255, cv2.FILLED)
    return img

def cut_background(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def get_board_and_background(img):
    contour = find_background(img)
    img_size = img.shape[:2]
    mask = np.zeros(img_size, dtype=np.uint8)
    contour_hull = cv2.convexHull(contour, False)
    # If the contour is not solid draw the covex hull instead
    if cv2.contourArea(contour) > 0.5 * cv2.contourArea(contour_hull):
        mask = draw_contours_on_img(mask, contour)
    else:
        mask = draw_contours_on_img(mask, contour_hull)
    return cut_background(img, mask), mask

def get_canny_edges(board):
    board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    low_threshold = 20
    high_threshold = 550
    edges = cv2.Canny(board_gray, low_threshold, high_threshold)
    
    # These transformations help fill in the hex edges such that watershed doesn't "leak"
    # into random nooks and crannies and form contours outside of the hexes
    edges = cv2.dilate(edges,None,iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    
    cv2.imshow("canny", edges)
    return edges

def get_markers(img):
    h, _, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    blue = [0.5 * 180, 0.65 * 180]
    markers = threshold_in_range(h, blue)
    return np.uint8(markers)

# Watershed algorithm
# https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
# Start with canny edges as "buckets," markers are the spouts that "fill"
# the buckets and segment each hex, to form a clean hexagonal grid
def get_watershed(canny, markers, background):
    unknown = cv2.subtract(background,markers)
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[unknown==255] = 0
    color_canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(color_canny, markers)
    ws = ws.astype(np.uint8)
    ws = cut_background(ws, background) # remove extra white line showing in bg
    cv2.imshow('ws', ws)
    return ws

def get_hex_grid(img):
    board, background = get_board_and_background(img)
    canny = get_canny_edges(board)
    markers = get_markers(board)
    hexes = get_watershed(canny, markers, background)
    return hexes

def overlay_grey_on_rgb(grey, rgb, color):
    grey_rgb = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)
    grey_rgb[grey==255] = color
    return cv2.addWeighted(rgb,.5,grey_rgb,.5,0)
    
# References:
# https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris#cornerharris
# https://www.geeksforgeeks.org/python-opencv-find-center-of-contour/
# Returns vertices circles, and list of vertices in sorted order according to our number scheme
def get_vertices(watershed_hexes):
    corner_probs = cv2.cornerHarris(watershed_hexes,21,23,0.04)

    # Result is dilated for marking the vertices roughly
    corner_probs = cv2.dilate(corner_probs,None,iterations=2)

    # Overlay hexagonal board onto original image
    # orig_img = cv2.subtract(orig_img, cv2.cvtColor(watershed_hexes, cv2.COLOR_GRAY2BGR))

    # Make a blank image
    rough_vertices = np.zeros(watershed_hexes.shape[:2], dtype='uint8')

    # Draw the vertices if the probability that it's a corner is high enough
    # Set the threshold for an optimal value, manually set
    threshold = 0.05280*corner_probs.max()
    rough_vertices[corner_probs>threshold] = 255
    cv2.imshow("rough_vertices", rough_vertices)

    # Now you have the rough_vertices, make them contours and find the centroid
    # Save all vertex centroids in a list
    contours, _ = cv2.findContours(rough_vertices, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    vertices_img = np.zeros(watershed_hexes.shape[:2], dtype='uint8')
    vertex_list = []
    for i in contours:
        M = cv2.moments(i)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            vertex_list.append((cx,cy))
            vertices_img = cv2.circle(vertices_img, (cx, cy), 7, 255, -1)
    cv2.imshow("vertex_circles", vertices_img)
    # Sort list by increasing y
    vertex_list.sort(key = lambda x: x[1], reverse=True)
    vertices = []
    for row_size in [2,4,6,6,6,6,6,6,6,4,2]:
        # Take one row at a time and sort them left to right.
        # If we simply sort by two keys (y then x), then if the points
        # are not exactly parallel there is a chance some rows are out of order
        row_list = []
        for i in range(row_size):
            if vertex_list:
                row_list.append(vertex_list.pop(0))
        row_list.sort(key = lambda x: x[0])
        vertices.extend(row_list)
    return vertices_img, vertices

vertex_list = None
project_id = "robotic-haven-256402"
topic_id = "test-topic"
subscription_id = "player_1"
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(project_id, topic_id)

def detect_piece(contour):
    return 's'

def find_and_send_moves(before, after):
    before, background = get_board_and_background(before)
    after = cut_background(after, background)
    # subtract to get difference
    diff =  cv2.subtract(before, after)
    # create grayscale of diff
    gray =  cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # create a mask for the non black values (above 10) 
    ret,thresh1 = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    # find contours in mask
    contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # calculate the center of each contour using the boundingrect
    print("Num contours: ", len(contours))
    data_str = ''
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        centerX = x+ int(w/2)
        centerY = y+ int(h/2)
        dists = []
        for i, v in enumerate(vertex_list):
            dists.append((dist((v[0],v[1]),(centerX,centerY)), i))
        dists.sort(key = lambda x: x[0])
        data_str += detect_piece(contours[0]) + ' ' + str(dists[0][1]) + ' '
        # Data must be a bytestring
    data = data_str.rstrip().encode("utf-8")
    if data_str:
        future = publisher.publish(topic_path, data)
        print(future.result())
        print(f"Published messages to {topic_path}.")
    cv2.imshow("diff", thresh1)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def main():
    # Initialize video stream
    vid = cv2.VideoCapture(1)
    time.sleep(1)

    # Take first image to initialize vertices data structures
    _, img = vid.read()
    hexes = get_hex_grid(img)
    cv2.imshow('hexes', hexes)
    vertices_img, vertices = get_vertices(hexes)
    global vertex_list
    vertex_list = vertices
    vertices_img = overlay_grey_on_rgb(vertices_img, img, [0,0,255])
    for i, v in enumerate(vertices):
        vertices_img = cv2.putText(vertices_img, str(i), (v[0], v[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow('vertices', vertices_img)
    cv2.waitKey(0)
    cv2.namedWindow("live video")
    cv2.namedWindow("before")
    cv2.namedWindow("after")
    # https://stackoverflow.com/questions/34588464/python-how-to-capture-image-from-webcam-on-click-using-opencv
    before, after = None, None
    is_before = True
    while True:
        ret, frame = vid.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("live video", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            if is_before:
                before = frame
                after = None
                cv2.imshow("before", before)
                print("before pic taken")
                is_before = False
            else:
                after = frame
                cv2.imshow("after", after)
                print("after pic taken")
                is_before = True
        if after is not None and before is not None:
            find_and_send_moves(before, after)
            after = None
            # this works just for the demo, otherwise reset to none
            # saves pressing a space to take another before pic
            before = after

    vid.release()
    

if __name__ == "__main__":
    main()

    