import cv2
import numpy as np

def count_cells(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges for red and white cells
    # These thresholds might need to be adjusted
    upper_red = np.array([338, 43, 68])
    lower_red = np.array([348, 17, 72])
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([180, 40, 255])

    # Threshold the HSV image to get only red and white colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours and count
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_white, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw contours on the original image (optional, for visualization)
    cv2.drawContours(image, contours_red, -1, (0, 255, 0), 2)
    cv2.drawContours(image, contours_white, -1, (255, 0, 0), 2)

    # Display the image and counts
    cv2.imshow('Detected Cells', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return len(contours_red), len(contours_white)

# Example usage
red_count, white_count = count_cells('Blood_Smear_Dataset/images/image-28.png')
print(f"Red blood cells: {red_count}")
print(f"White blood cells: {white_count}")
