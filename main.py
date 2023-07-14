import cv2
import numpy as np
import tensorflow as tf

# Load the trained model from the H5 file
model = tf.keras.models.load_model('eye_model.h5')

# Load the eye cascade classifier XML file
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera, or provide the path to a video file

while True:
    # Read the current frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform eye detection
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate over detected eyes
    for (x, y, w, h) in eyes:
        # Extract the eye region from the frame
        eye_image = frame[y:y+h, x:x+w]

        # Preprocess and resize the eye image
        resized_eye = cv2.resize(eye_image, (64, 64))
        normalized_eye = resized_eye.astype('float32') / 255.0
        input_eye = np.expand_dims(normalized_eye, axis=0)

        # Make predictions for the eye image
        predictions = model.predict(input_eye)

        # Get the predicted class
        predicted_class = np.argmax(predictions[0])

        # Map the predicted class index to the actual class label
        class_labels = ['close', 'center', 'left', 'right']
        predicted_label = class_labels[predicted_class]

        # Draw a rectangle around the detected eye
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the predicted label on the frame
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the output label within the eye region
        cv2.putText(frame, predicted_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame with detections and predictions
    cv2.imshow('Video', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
