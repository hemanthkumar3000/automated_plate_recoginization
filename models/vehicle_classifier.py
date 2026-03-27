import numpy as np
import cv2
import os

class VehicleClassifier:
    """Vehicle classification using Keras model"""
    
    def __init__(self, model_path='models/keras_Model.h5', labels_path='models/labels.txt', mock_mode=False):
        self.model = None
        self.class_names = []
        self.mock_mode = mock_mode
        
        if not mock_mode:
            try:
                from keras.models import load_model
                if os.path.exists(model_path):
                    self.model = load_model(model_path, compile=False)
                    if os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            self.class_names = [line.strip() for line in f.readlines()]
                    print(f"✓ Loaded classifier with {len(self.class_names)} classes")
                else:
                    print(f"⚠ Model not found at {model_path}, using mock classifier")
                    self.mock_mode = True
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                self.mock_mode = True
        
        if self.mock_mode:
            self.class_names = ['Car', 'Bike', 'Truck', 'Bus', 'Auto']
            print("✓ Using mock classifier with default classes")
    
    def classify_image(self, image):
        """Classify vehicle type from image"""
        if self.mock_mode:
            # Mock classification based on image properties
            h, w = image.shape[:2]
            aspect = w / h
            
            if aspect > 1.5:
                return "Car", 0.75
            elif aspect < 0.8:
                return "Bike", 0.70
            elif h > 400:
                return "Bus", 0.65
            else:
                return "Truck", 0.60
        
        try:
            # Resize and preprocess
            resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image_array = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            
            # Predict
            prediction = self.model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index] if self.class_names else "Vehicle"
            confidence = float(prediction[0][index])
            
            # Clean class name
            if isinstance(class_name, str) and ' ' in class_name:
                class_name = class_name.split(' ')[-1]
            
            return class_name, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return "Unknown", 0.0