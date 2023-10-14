import cv2

# Load the pre-trained palm cascade classifier
palm_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'palm.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)  # Change the index to 1 if you're using a USB camera

while True:
    # Read the current frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect palms in the frame
    palms = palm_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw rectangles around the detected palms
    for (x, y, w, h) in palms:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Detected Palm', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with detected palms
    cv2.imshow('Palm Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
