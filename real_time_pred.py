import cv2
import tensorflow as tf
import numpy as np

# Load the pretrained model
model = tf.keras.models.load_model('mnist_cnn.h5')

# Initialize the camera
# 0 refers to the default camera, 1 the next one available, and so on
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Binarize the image (you might need to adjust the threshold value)
    _, binarized = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # You might want to detect contours or regions of interest, but for simplicity, let's resize the entire frame
    resized_digit = cv2.resize(binarized, (28, 28))
    normalized_digit = resized_digit.astype('float32') / 255.0
    normalized_digit = np.expand_dims(normalized_digit, axis=-1)
    normalized_digit = np.expand_dims(normalized_digit, axis=0)

    # Predict the number
    prediction = model.predict(normalized_digit)
    predicted_number = np.argmax(prediction)

    # Display the frame and prediction
    cv2.putText(frame, str(predicted_number), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('Frame', frame)

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture stream and close windows
cap.release()
cv2.destroyAllWindows()
