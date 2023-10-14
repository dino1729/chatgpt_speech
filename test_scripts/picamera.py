import cv2

# Load the pre-trained fist cascade classifier
fist_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'fist.xml')

# Initialize the camera
camera = cv2.VideoCapture(0)  # Change the index to 1 if you're using a USB camera

while True:
    # Read the current frame from the camera
    ret, frame = camera.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect fists in the frame
    fists = fist_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected fists
    for (x, y, w, h) in fists:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Detected fist', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print("fist detected!")

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
