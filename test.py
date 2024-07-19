import cv2


image_path = 'Blood_Smear_Dataset/images/image-28.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Adaptive Thresholding
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

# Draw contours
cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

# Display the image with contours
cv2.imshow('Cells Detected', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the count
print(f"Detected cells: {len(filtered_contours)}")
