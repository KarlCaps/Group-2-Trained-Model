# Group-2-Trained-Model
An image classification machine learning model that identifies pens, pencils, markers, and paint brushes from uploaded images using Google Teachable Machine and TensorFlow.js.

# Pen, Pencil, Marker, or Paint Brush Classifier

## About the Project

This project is an image classification machine learning model designed to identify whether an uploaded or attached image contains a Pen, Pencil, Marker, or Paint Brush.

The model is trained using Google Teachable Machine and can be integrated into a web application using TensorFlow.js. Users can upload an image, and the model analyzes it to predict which of the four classes the object belongs to.

## Objectives

- Learn the fundamentals of image classification and machine learning.
- Train a model to recognize four different writing and painting tools.
- Allow users to upload images for classification.
- Display the predicted class based on the model's output.
- Explore how machine learning can be integrated into a web application.

## Classification Classes

The model is trained to recognize the following classes:

1. **Pen** – An instrument commonly used for writing with ink.
2. **Pencil** – An instrument commonly used for writing or drawing with graphite or colored cores.
3. **Marker** – A writing or drawing instrument that uses ink and usually has a broad or felt tip.
4. **Paint Brush** – A tool with bristles used to apply paint to surfaces.

## Technologies Used

- **Google Teachable Machine** – For collecting training images and training the image classification model.
- **TensorFlow.js** – For running the trained machine learning model in a web browser.
- **HTML** – For the structure of the web application.
- **CSS** – For the design and layout of the interface.
- **JavaScript** – For loading the model, processing uploaded images, and displaying predictions.

## How It Works

1. Collect images of pens, pencils, markers, and paint brushes.
2. Organize the images into their respective classes in Google Teachable Machine.
3. Train the image classification model.
4. Export the trained model for use with TensorFlow.js.
5. Load the model in the web application.
6. Upload an image of a writing or painting tool.
7. The model analyzes the image and predicts the most likely class.
8. Display the predicted result to the user.

## How to Use

1. Clone or download this repository.
2. Make sure the exported model files and TensorFlow.js library are available in the correct locations.
3. Open the project using a local development server if required.
4. Open the web application in a supported browser.
5. Upload an image containing a pen, pencil, marker, or paint brush.
6. View the model's predicted class and confidence score, if implemented.

## Model Limitations

- Prediction accuracy depends on the quality and variety of the training images.
- Poor lighting, unusual angles, blurry images, and cluttered backgrounds may affect predictions.
- Similar-looking objects may be misclassified.
- The model is designed to classify the four trained classes and may not correctly identify objects outside those classes.
- Confidence scores represent the model's prediction, not a guarantee that the result is correct.

## Future Improvements

- Add more training images with different backgrounds, lighting conditions, and angles.
- Improve the model using more diverse and balanced training data.
- Add a clearer interface for displaying predictions and confidence scores.
- Test the model using images that were not included in the training dataset.
- Improve handling of unsupported or unrelated images.

## Purpose

This project is developed for educational purposes to demonstrate image classification, machine learning model training, and web-based AI integration.

## Acknowledgment

This project uses Google Teachable Machine for model training and TensorFlow.js for browser-based machine learning integration.
