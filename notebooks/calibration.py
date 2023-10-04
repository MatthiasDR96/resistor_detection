# Imports
import cv2
import yaml
import numpy

# File name
file = '../data/demo3_hsv_resistor.npy'

# Read data from previous calibrations
hsvfile = numpy.load(file)

def nothing(*args):
    pass

# Make window
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Calibration', 1902, 1280)

# Make sliders
cv2.createTrackbar('Hmin', 'Calibration', hsvfile[0], 179, nothing)
cv2.createTrackbar('Hmax', 'Calibration', hsvfile[1], 179, nothing)
cv2.createTrackbar('Smin', 'Calibration', hsvfile[2], 255, nothing)
cv2.createTrackbar('Smax', 'Calibration', hsvfile[3], 255, nothing)
cv2.createTrackbar('Vmin', 'Calibration', hsvfile[4], 255, nothing)
cv2.createTrackbar('Vmax', 'Calibration', hsvfile[5], 255, nothing)
cv2.createTrackbar('save', 'Calibration', 0, 1, nothing)

# Define image formats
HSVmin = numpy.zeros((1080, 1920, 3), numpy.uint8)
HSVmax = numpy.zeros((1080, 1920, 3), numpy.uint8)
HSVgem = numpy.zeros((1080, 1920, 3), numpy.uint8)
white_image = numpy.zeros((1080, 1920, 3), numpy.uint8)

# Initial mask
white_image[:] = [255, 255, 255]

# Loop
while True:

    # Get slider values
    hmin = cv2.getTrackbarPos('Hmin', 'Calibration')
    hmax = cv2.getTrackbarPos('Hmax', 'Calibration')
    smin = cv2.getTrackbarPos('Smin', 'Calibration')
    smax = cv2.getTrackbarPos('Smax', 'Calibration')
    vmin = cv2.getTrackbarPos('Vmin', 'Calibration')
    vmax = cv2.getTrackbarPos('Vmax', 'Calibration')
    save = cv2.getTrackbarPos('save', 'Calibration')

    # Read images
    color_image = cv2.imread('../data/resistor_images/1_zkk.jpg')

    # Define bounds on Hue value
    lower_color = numpy.array([hmin, smin, vmin])
    upper_color = numpy.array([hmax, smax, vmax])

    # Gaussian blur
    blurred_image = cv2.GaussianBlur(color_image, (7, 7), 0)

    # Convert to hsv color space
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Get mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Erode to close gaps
    mask = cv2.erode(mask, None, iterations=2)

    # Dilate to reduce data
    mask = cv2.dilate(mask, None, iterations=2)

    # Apply mask to image
    res = cv2.bitwise_and(color_image, color_image, mask=mask)

    # Binary of image
    mask_bgr = cv2.bitwise_and(white_image, white_image, mask=mask)

    # Mount all images
    img = numpy.hstack((color_image, mask_bgr, res))

    # Show image
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Calibration', 1920, 1080)
    cv2.imshow('Calibration', img)
    cv2.waitKey(1)

    # Leave loop on save button
    if (save): break

# Save data
hsvarray = numpy.array([hmin, hmax, smin, smax, vmin, vmax])
numpy.save(file, hsvarray)