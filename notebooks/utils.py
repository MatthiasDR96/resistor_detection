# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_color_bands(image):

	# Convert to gray to threshold background
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Threshold background
	_, threshed = cv2.threshold(image_gray, 254, 255, cv2.THRESH_BINARY_INV)

	# The kernel is chosen to be larger than the sticks, and smaller than the resistor
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

	# We open the image in order to remove the sticks
	morphed_open = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, kernel)

	# Find contour of resistor
	contours = cv2.findContours(morphed_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Get largest contour
	maxcontour = max(contours, key=cv2.contourArea)

	# Get minimal area rectangle
	rect = cv2.minAreaRect(maxcontour)

	# Get rectangle properties
	angle = rect[2]
	rows, cols = image.shape[0], image.shape[1]

	# Rotate image
	M = cv2.getRotationMatrix2D((rect[0][0],rect[0][1]), angle-90, 1)
	image_aligned = cv2.warpAffine(image,M,(cols,rows))

	# Rotate bounding box 
	box = cv2.boxPoints((rect[0], rect[1], angle))
	pts = np.intp(cv2.transform(np.array([box]), M))[0]    
	pts[pts < 0] = 0

	# Cropping
	image_cropped = image_aligned[pts[0][1]:pts[3][1], pts[0][0]:pts[2][0]]

	# Get HSV calibration params 
	hsvfile1 = np.load('../data/demo3_hsv_resistor.npy')
	hsvfile2 = np.load('../data/demo3_hsv_background.npy')

	# Convert image to HSV
	hsv = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2HSV)

	# Remove area in between color bands
	mask1 = cv2.bitwise_not(cv2.inRange(hsv, np.array([hsvfile1[0], hsvfile1[2], hsvfile1[4]]), np.array([hsvfile1[1], hsvfile1[3], hsvfile1[5]])))
	mask2 = cv2.inRange(hsv, np.array([hsvfile2[0], hsvfile2[2], hsvfile2[4]]), np.array([hsvfile2[1], hsvfile2[3], hsvfile2[5]]))
	mask = cv2.bitwise_and(mask1, mask2)

	# Morphological transformations to remove sticks
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	morphed_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel)
	
	# Find the three largest contours of the color bands
	contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

	# Get three largest contours
	largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:3]

	# Sort contours from left to right
	sorted_contours = sorted(largest_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

	# Iterate over three contours
	color_bands = []
	for ctr in sorted_contours:

		# Get roi
		x,y,w,h = cv2.boundingRect(ctr)
		roi = hsv[y+10:y+h-10, x+5:x+w-5]

		# Get hsv
		roi_h = [i for i in roi[:,:,0].ravel() if i != 0]  
		roi_s = [i for i in roi[:,:,1].ravel() if i != 0]  
		roi_v = [i for i in roi[:,:,2].ravel() if i != 0]  

		# Get means of HSV data
		mean_hsv = [np.mean(roi_h), np.mean(roi_s), np.mean(roi_v)]

		# Predict
		color_bands.append(mean_hsv)

	return color_bands

def decode(bands):

    # Init
    color = ['k', 'z', 'r', 'o', 'y','g', 'b', 'v', 'x', 'w'] # [black, brown, red, orange, yellow, green, blue, violet, gray, white]
    value = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ohm, multiplier, unit = '', '', ' ohms'

    # Decode third band
    for i in range(len(color)):
        if bands[2] == color[i]:
            multiplier = 10**value[i]

    # Decode secodn and first band
    for i in range(len(color)):
        if bands[1] == color[i]:
            bands_one = value[i]
        if bands[0] == color[i]:
            bands_zero = value[i]

    # Calculate result
    res = int(str(bands_zero) + str(bands_one))*multiplier

    # Convert
    if res >= 1000000:
        res = str(res/1000000)
        if res[-1] == '0':
            res = res[0:-2] + 'M'
        else: res = res + 'M'
    elif res >= 1000:
        res = str(res/1000)
        if res[-1] == '0':
            res = res[0:-2] + 'k'
        else: res = res + 'k'

    # Finish
    ohm = str(res) + unit 

    return ohm