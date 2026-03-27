import cv2
from utils.anpr import ANPRProcessor
import logging

class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.anpr = ANPRProcessor(mock_mode=mock_mode)
    
    def classify_video(self, video_path, classifier):
        """Process video and classify vehicles"""
        result = {
            'success': False,
            'vehicle_type': 'Unknown',
            'plate_number': None,
            'confidence': 0.0,
            'frames_processed': 0
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error'] = "Could not open video file"
                return result
            
            frame_count = 0
            predictions = []
            plates_detected = set()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every 30th frame
                if frame_count % 30 == 0:
                    # Classify vehicle type
                    if classifier:
                        vehicle_type, conf = classifier.classify_image(frame)
                        if vehicle_type:
                            predictions.append((vehicle_type, conf))
                    
                    # Detect plate
                    plate_result = self.anpr.detect_plate(frame)
                    if plate_result['success']:
                        plates_detected.add(plate_result['plate_number'])
                    
                    result['frames_processed'] += 1
            
            cap.release()
            
            # Get most common prediction
            if predictions:
                from collections import Counter
                most_common = Counter([p[0] for p in predictions]).most_common(1)[0]
                result['vehicle_type'] = most_common[0]
                result['confidence'] = max([p[1] for p in predictions if p[0] == most_common[0]])
                result['success'] = True
            
            # Get plate number
            if plates_detected:
                result['plate_number'] = list(plates_detected)[0]
            
            if not result['success']:
                result['vehicle_type'] = "Vehicle"
                result['confidence'] = 0.5
                result['success'] = True
            
            return result
            
        except Exception as e:
            logging.error(f"Video processing error: {e}")
            result['error'] = str(e)
            return result