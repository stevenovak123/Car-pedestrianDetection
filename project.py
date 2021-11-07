# haar features. Black and white blocks to represent the car winshield and computes with white and black pixels and helps train the data.
# importing openCV library
import cv2

# importing Target video
video = cv2.VideoCapture('video2.mp4')
# video = cv2.VideoCapture('video1.mp4')
# video = cv2.VideoCapture('video3.mp4')
# video = cv2.VideoCapture('video4.mp4')

# classifier receives training data that labels images.Basically the skeletons.
# Here we are using pre-trained values.

# The xml file contains all the possible car co-ordinates post training the model.
car_file = "cars.xml"

# pedestrian classifier
pedestrian_file = "pedestrian.xml"


# creating the classifier. It will reference the file and check with the frame
car_tracker = cv2.CascadeClassifier(car_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)


while True:
    # Read the video frames
    (read, frame) = video.read()

    if read:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

# detect cars & pedestrians in the video frame by frame
    cars = car_tracker.detectMultiScale(grayscale)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)


# Converts all the pixels into array of arrays to each pixel has its own value.

# Display
    cv2.imshow('Car and Pedestrian Detector', frame)

# using waitKey to make sure that the window does not auto shut once it is done and 
# wait for a keypress
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

video.release()
print("End of code")
