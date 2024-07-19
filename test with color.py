import cv2
import numpy as np


def segment_cells(image, lower_color, upper_color):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    return mask


def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Define color ranges in HSV
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([180, 40, 255])

    # Segment red and white blood cells
    mask_red = segment_cells(image, lower_red, upper_red)
    mask_white = segment_cells(image, lower_white, upper_white)

    # Find contours for red cells
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_count = len([c for c in contours_red if cv2.contourArea(c) > 50])

    # Find contours for white cells
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    white_count = len([c for c in contours_white if cv2.contourArea(c) > 50])

    # Output results
    print(f"Red blood cells: {red_count}")
    print(f"White blood cells: {white_count}")

    # Optional: Visualize the results
    cv2.drawContours(image, contours_red, -1, (0, 255, 0), 2)  # Red cells in green contours
    cv2.drawContours(image, contours_white, -1, (0, 0, 255), 2)  # White cells in blue contours
    cv2.imshow('Detected Cells', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to your image
image_path = 'Blood_Smear_Dataset/images/image-28.png'
process_image(image_path)
