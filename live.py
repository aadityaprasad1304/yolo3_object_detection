import cv2
import numpy as np

# Load YOLO files - YOLOv3 weights and configuration
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names from COCO file
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Load video from the default camera (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Break the loop if there is an issue reading the frame
    if not ret:
        break

    # Convert the frame to a blob that YOLO can process
    # Parameters: frame, scale factor, size, mean subtraction values, swap channels, crop
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input blob for the neural network
    net.setInput(blob)

    # Forward pass the input blob through the network to get detections
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process YOLO output
    for out in outs:
        for detection in out:
            # Extract class scores, class ID, and confidence from the detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections by confidence threshold (e.g., 0.5)
            if confidence > 0.5:
                # Calculate bounding box coordinates
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow("Live Analysis", frame)
    
    # Quit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture resources
cap.release()

# Close OpenCV windows
cv2.destroyAllWindows()
