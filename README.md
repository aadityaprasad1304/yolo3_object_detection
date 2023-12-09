# Real-Time Object Detection with YOLO and OpenCV


## Overview

This project showcases real-time object detection using the YOLO (You Only Look Once) deep learning model in conjunction with OpenCV. The YOLO model is pre-trained on the COCO (Common Objects in Context) dataset, enabling it to identify various objects in real-time video streams.

## Features

- **Live Object Detection:** Utilizes the YOLOv3 model to perform object detection on a live video feed from the default camera.
- **Bounding Boxes and Labels:** Detected objects are highlighted with bounding boxes and labeled with class names and confidence scores.
- **Configurable Confidence Threshold:** Adjust the confidence threshold to filter detections based on confidence scores.
- **Easy to Use:** Minimal setup and dependencies for quick deployment.

## Dependencies

- OpenCV
- NumPy

## Project Structure

- `your_script_name.py`: The main Python script for real-time object detection.
- `coco.names`: File containing the names of classes from the COCO dataset.
- `yolov3.weights`: Pre-trained YOLOv3 weights.
- `yolov3.cfg`: YOLOv3 model configuration file.

## Setup Instructions

1. **Download COCO class names file:**
    ```bash
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    ```

2. **Download YOLOv3 weights:**
    - Full weights:
        ```bash
        wget https://pjreddie.com/media/files/yolov3.weights
        ```

3. **Download YOLOv3 configuration file (yolov3.cfg).**

4. **Install Python dependencies:**
    ```bash
    pip install opencv-python numpy
    ```

5. **Run the Python script:**
    ```bash
    python live.py
    ```

## Usage

- The script captures video from the default camera (index 0).
- Detected objects with confidence scores greater than the specified threshold (default: 0.5) are displayed with bounding boxes and labels.
- Press 'q' to quit the application.

## Configuration

- Adjust parameters in the script such as confidence thresholds, camera index, etc., as needed for your use case.

## Customization

- Extend the functionality by integrating it with other projects or frameworks.
- Customize the script to use different YOLO versions or models based on your requirements.

## Contribution Guidelines

Contributions are welcome! If you find any issues or have enhancements to suggest, please open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The YOLO model is developed by Joseph Redmon and maintained by the [Darknet](https://github.com/pjreddie/darknet) community.
- The COCO dataset is maintained by the [COCO Consortium](https://cocodataset.org/).

## Contact

For inquiries and suggestions, please contact [Your Name] at [your.email@example.com].

