import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('mnist_cnn.h5')

# Load and preprocess the image
image = cv2.imread('image\mnist test.png', cv2.IMREAD_GRAYSCALE)
_, binarized = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(
    binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract bounding boxes and sort by their y-coordinate (top to bottom)
bounding_boxes = [cv2.boundingRect(c) for c in contours]
sorted_bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

predicted_numbers = []

for x, y, w, h in sorted_bounding_boxes:
    # Extract each digit
    digit = image[y:y+h, x:x+w]

    # Resize and preprocess for the model
    resized_digit = cv2.resize(digit, (28, 28))
    normalized_digit = resized_digit.astype('float32') / 255.0
    normalized_digit = np.expand_dims(normalized_digit, axis=-1)
    normalized_digit = np.expand_dims(normalized_digit, axis=0)

    # Predict the number
    prediction = model.predict(normalized_digit)
    predicted_number = np.argmax(prediction)
    predicted_numbers.append(predicted_number)

# Group numbers by rows
final_numbers = []
current_row = y
number = ""

for x, y, w, h in sorted_bounding_boxes:
    if abs(y - current_row) > h:  # New line
        if number:  # Check if the number string is not empty
            final_numbers.append(int(number))
        number = ""
        current_row = y
    prediction = model.predict(np.expand_dims(
        cv2.resize(image[y:y+h, x:x+w], (28, 28)), axis=[0, -1]))
    number += str(np.argmax(prediction))

if number:  # Check for the last number string too
    final_numbers.append(int(number))

print(final_numbers)


# Convert each chunk into a two-digit number and append to a list
# final_numbers = [int(''.join(map(str, pair)))
# for pair in paired_numbers if len(pair) == 2]


# print(final_numbers)

contours, _ = cv2.findContours(
    binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
debug_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 2)
cv2.imshow("Segmented Digits", debug_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
