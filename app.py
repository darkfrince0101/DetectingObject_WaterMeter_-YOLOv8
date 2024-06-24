from flask import Flask, redirect, url_for, render_template, request, send_from_directory
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from sklearn.decomposition import PCA
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from PIL import Image
import shutil

app = Flask(__name__)

# Define directories for storing different types of images
UPLOAD_FOLDER = 'uploads'
CROP_FOLDER = 'crop_image'
NUMBER_FOLDER = 'number_image'
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(NUMBER_FOLDER, exist_ok=True)

# Configuration directories for Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CROP_FOLDER'] = CROP_FOLDER
app.config['NUMBER_FOLDER'] = NUMBER_FOLDER

# allow type of input(Image)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load YOLO models
meter_model = YOLO('model/MeterDetection.pt')
number_model = YOLO('model/NumberDetection.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# configure value return for homepage
@app.route("/")
def home():
    # Retrieve filenames from request arguments
    filename = request.args.get('filename', None)
    crop_filename = request.args.get('crop_filename', None)
    number_filename =  request.args.get('number_filename', None)
    meter_result = request.args.get('meter_result', None)
    return render_template("index.html", filename=filename, crop_filename=crop_filename, number_filename=number_filename, meter_result=meter_result, error_message=request.args.get('error_message'))

@app.route('/upload', methods=['POST'])
# main function for process input water meter image
def upload_file():
    # if there is not have input image 
    if 'image' not in request.files:
        return redirect(url_for('home'))
    
    # save the input image in the index.html to folder uploads
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Remove the folder runs from the process of the model to avoid conflict directories
    if os.path.exists('runs'):
        shutil.rmtree('runs')

    # process the image in the uploads folder
    img = cv2.imread(filepath)

    try:
        # run the model to predict the Meter to crop 
        results = meter_model.predict(source=img, conf=0.8, save_crop = True)
        
        # Define the crop image which the model predict
        crop_src_path = os.path.join('runs/detect/predict/crops/Meter/', f'image0.jpg')

        # Set name of the crop image 
        crop_filename = "crop_"+filename

        # Define new path to save the crop image
        crop_dest_path = os.path.join(app.config['CROP_FOLDER'], crop_filename)

        # Check if the crop image's name exist, delete it 
        if os.path.exists(crop_dest_path):
            os.remove(crop_dest_path)

        # Move the crop image from default folder to new directory
        if os.path.exists(crop_src_path):
            os.rename(crop_src_path, crop_dest_path)
        else:
            raise FileNotFoundError("Crop image was not generated.")
    except Exception as e:
        # Handle any errors during the crop image generation process
        print(f"Error during image cropping: {e}")
        return redirect(url_for('home',filename = filename, error_message="Model cannot detect the Meter"))

    # Process the crop image
    detect_img = cv2.imread(crop_dest_path)

    # Get the angle to rotate the image to make the number horizon 
    angle = detect_orientation(detect_img)
    print("angle: ", angle)

    # rotate the crop image and read the number from left to right
    rotated_image = rotate_image(detect_img, angle)
    meter_result = Meter_reading(rotated_image)
    
    # Define the labeled number image which the model predict
    number_src_path = os.path.join('runs/detect/predict2/', f'image0.jpg') 

    # Set name of the labeled number image 
    number_filename = "number_"+filename
    
    # Define new path to save the labeled number image
    number_dest_path = os.path.join(app.config['NUMBER_FOLDER'], number_filename)

    # Check if the labeled number image's name exist, delete it 
    if os.path.exists(number_dest_path):
            os.remove(number_dest_path)

    # Move the labeled number image from default folder to new directory
    if os.path.exists(number_src_path):
            os.rename(number_src_path, number_dest_path)

    # return all the values to homepage
    return redirect(url_for('home', filename=filename, crop_filename= crop_filename, number_filename = number_filename,meter_result=meter_result))

# Create route to access the image in uploads folder 
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Create route to access the image in crop folder 
@app.route('/crops/<filename>')
def cropped_file(filename):
    return send_from_directory(app.config['CROP_FOLDER'], filename)

# Create route to access the image in number folder 
@app.route('/numbers/<filename>')
def numbered_file(filename):
    return send_from_directory(app.config['NUMBER_FOLDER'], filename)

# Function to predict orientation by Hough algorithm
def hough_transforms(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Apply Canny edge detector
    edges = canny(blurred, sigma=3)

    # Define the range of angles to be tested
    tested_angles = np.deg2rad(np.arange(0, 180, 0.1))

    # Perform the Hough transform
    h, theta, d = hough_line(edges, theta=tested_angles)

    # Extract peaks from the Hough transform
    accum, angles, dists = hough_line_peaks(h, theta, d)

    # Interpret angles to determine the correction needed
    if len(angles) > 0:
        # Calculate the predominant angle
        predominant_angle = np.rad2deg(np.mean(angles))
        print('predominant_angle: ', predominant_angle)
        # Correcting for alignment to horizontal
        if predominant_angle < 45 or predominant_angle > 135:
            return -90 + predominant_angle  # Aligning vertical lines to be horizontal
        else:
            return -predominant_angle      # Minimal correction towards horizontal
    else:
        return 0

# function to return the angle of present crop image
def detect_orientation(image):
    if image.size == 0:
        raise ValueError("Empty image provided")
    #check if the vertical image to horizontal image
    height, width = image.shape[:2]
    if height > 2*width:
      return hough_transforms(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find all non-zero points
    coordinates = np.column_stack(np.where(binary > 0))

    # Check if coordinates are empty
    if coordinates.size == 0:
        raise ValueError("No features detected to analyze in the image")

    # Perform PCA to find the principal component
    pca = PCA(n_components=1)
    pca.fit(coordinates)
    first_component = pca.components_[0]

    # Calculate the angle in radians and convert to degrees
    angle = np.arctan2(first_component[1], first_component[0])
    adjusted_angle = np.degrees(angle)

    # Adjust angle to keep the text horizontal
    # Assuming horizontal readability is desired (e.g., for meter reading)
    if -45 < adjusted_angle <= 45:
        # The text is likely already horizontal
        return adjusted_angle
    elif 45 < adjusted_angle <= 135:
        # Text runs from bottom to top, rotate left
        return adjusted_angle - 90
    elif -135 <= adjusted_angle < -45:
        # Text runs from top to bottom, rotate right
        return adjusted_angle + 90
    else:
        # Text may be upside down
        return adjusted_angle + 180 if adjusted_angle <= 0 else adjusted_angle - 180

#function to rotate the image
def rotate_image(image, angle):
    # Get the image dimensions (height, width)
    height, width = image.shape[:2]

    # Point to rotate around: the center of the image
    center = (width // 2, height // 2)

    # Rotation matrix: compute the rotation matrix around the center
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute the new bounding dimensions of the image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the actual rotation and return the image
    return cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

#Main function to return number on meter
def Meter_reading(rotated_image):

    image = rotated_image

    # Run the model to predict the number meter from the rotated image
    predictions = number_model.predict(source=image,save = True)

    # Accessing the detected boxes
    detections = predictions[0].boxes  

    # Access the necessary attributes from detections
    boxes = detections.xyxy  # Bounding boxes in xyxy format
    confidences = detections.conf  # Confidence values for each box
    class_ids = detections.cls  # Class indices

    # Filter detections based on a confidence threshold
    confidence_threshold = 0.2
    filtered_indices = [i for i, conf in enumerate(confidences) if conf > confidence_threshold]
    filtered_boxes = [boxes[i] for i in filtered_indices]
    filtered_class_ids = [class_ids[i] for i in filtered_indices]

    # Convert class indices to actual number names using the 'names' dictionary in predictions[0]
    detected_numbers_with_coords = [(predictions[0].names[int(cls)], box[0]) for cls, box in zip(filtered_class_ids, filtered_boxes)]

    # Sort by the x-coordinate (horizontal position)
    detected_numbers_with_coords.sort(key=lambda x: x[1])

    # Extract sorted detected numbers
    sorted_detected_numbers = [num for num, _ in detected_numbers_with_coords]

    result = "".join(sorted_detected_numbers)

    return result

if __name__ == "__main__":
    app.run(debug=True)