import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from picamera2 import Picamera2
from libcamera import controls
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import time
import datetime
import threading

class FoodDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Food Calorie Calculator")
        
        # Load model and labels
        self.model = self.load_model()
        self.labels = self.load_labels()
        self.food_db = self.load_food_database()
        
        # Initialize camera
        self.cam = Picamera2()
        self.cam_config = self.cam.create_still_configuration(
            main={"size": (640, 480)}  # Set desired resolution
        )
        self.cam.configure(self.cam_config)
        
        # GUI setup
        self.setup_gui()
        
    def load_model(self):
        model_filepath = "/home/pi/Downloads/model.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_filepath)
        interpreter.allocate_tensors()
        return interpreter
    
    def load_labels(self):
        with open("labels.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    
    def load_food_database(self):
        return {
            "Apple": {"calories": 52, "healthy": True},
            "Banana": {"calories": 89, "healthy": True},
            "Burger": {"calories": 313, "healthy": False},
            "Chocolate": {"calories": 535, "healthy": False},
            "French Fries": {"calories": 312, "healthy": False},
            "Rice": {"calories": 130, "healthy": True}
        }
    
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        camera_frame = ttk.LabelFrame(main_frame, text="Food Detection", padding="10")
        camera_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.camera_btn = ttk.Button(camera_frame, text="Open Camera Preview", command=self.start_camera_preview)
        self.camera_btn.grid(row=0, column=0, pady=5)
        
        self.result_label = ttk.Label(camera_frame, text="Detected food: None")
        self.result_label.grid(row=1, column=0, pady=5)

        self.confidence_label = ttk.Label(camera_frame, text="Confidence: 0%")
        self.confidence_label.grid(row=2, column=0)
        
        # Portion details setup...
    
    def start_camera_preview(self):
        self.cam.start()
        self.capture_image()
    
    def capture_image(self):
        rgb_array = self.cam.capture_array()
        image = Image.fromarray(rgb_array)
        image.show()  # This will pop up the image
        
        # Process the captured image
        detected_food, confidence = self.detect_food(image)
        if detected_food:
            self.result_label.config(text=f"Detected food: {detected_food}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        else:
            self.result_label.config(text="No food detected")
            self.confidence_label.config(text="")
    
    def detect_food(self, image):
        input_shape = self.model.get_input_details()[0]['shape'][1:3]
        image = image.resize(input_shape)
        input_array = np.array(image, dtype=np.float32)[np.newaxis, :, :, :]
        input_array = input_array[:, :, :, (2, 1, 0)]  # Convert to BGR
        
        self.model.set_tensor(self.model.get_input_details()[0]['index'], input_array)
        self.model.invoke()
        
        outputs = self.model.get_tensor(self.model.get_output_details()[0]['index'])
        max_index = np.argmax(outputs[0])
        tag = self.labels[max_index]
        probability = outputs[0][max_index]
        
        if probability < 0.5:
            return None, 0.0
            
        return tag, probability

    def cleanup(self):
        self.cam.stop()
        if hasattr(self, 'cam'):
            self.cam.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FoodDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()
