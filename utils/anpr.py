import cv2
import numpy as np
import pytesseract
import re
import logging

# Set Tesseract path (adjust if needed)
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass

class ANPRProcessor:
    """Number Plate Recognition Processor"""
    
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.plate_patterns = [
            r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}',
            r'[A-Z]{2}\d{2}[A-Z]{1}\d{4}',
            r'[A-Z]{2}\d{2}[A-Z]{2}\d{3}',
            r'[A-Z]{1,2}\d{1,2}[A-Z]{1,2}\d{3,4}',
        ]
        
        if mock_mode:
            print("✓ ANPR running in mock mode")
    
    def preprocess_plate(self, plate_image):
        """Preprocess plate image for OCR"""
        try:
            if len(plate_image.shape) == 3:
                gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = plate_image
            
            # Resize
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Threshold
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.medianBlur(thresh, 3)
            
            return denoised
        except Exception as e:
            logging.error(f"Preprocess error: {e}")
            return plate_image
    
    def extract_plate_number(self, text):
        """Extract number plate from OCR text"""
        text = text.upper().replace(' ', '').replace('\n', '')
        
        for pattern in self.plate_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]
        
        # Fallback: find alphanumeric strings
        alnum_matches = re.findall(r'[A-Z0-9]{6,10}', text)
        for match in alnum_matches:
            if re.search(r'[A-Z]', match) and re.search(r'\d', match):
                if 6 <= len(match) <= 10:
                    return match
        
        return None
    
    def detect_plate_region(self, image):
        """Detect number plate region in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                area = cv2.contourArea(contour)
                
                if 2 < aspect_ratio < 6 and w > 60 and h > 20 and area > 1000:
                    return (x, y, w, h)
            
            return None
        except Exception as e:
            logging.error(f"Plate detection error: {e}")
            return None
    
    def detect_plate(self, image):
        """Main plate detection function"""
        result = {
            'success': False,
            'plate_number': None,
            'image': image.copy(),
            'confidence': 0.0
        }
        
        if self.mock_mode:
            # Mock detection - simulate finding a plate
            h, w = image.shape[:2]
            x, y = int(w*0.3), int(h*0.6)
            plate_w, plate_h = int(w*0.4), int(h*0.1)
            
            cv2.rectangle(result['image'], (x, y), (x+plate_w, y+plate_h), (0, 255, 0), 3)
            mock_plate = "AP39MG5914"
            cv2.putText(result['image'], mock_plate, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            result['success'] = True
            result['plate_number'] = mock_plate
            result['confidence'] = 0.85
            return result
        
        try:
            # Detect plate region
            plate_box = self.detect_plate_region(image)
            
            if plate_box:
                x, y, w, h = plate_box
                plate_region = image[y:y+h, x:x+w]
                
                # Preprocess
                processed = self.preprocess_plate(plate_region)
                
                # OCR
                custom_config = r'--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(processed, config=custom_config)
                
                # Extract plate number
                plate_number = self.extract_plate_number(text)
                
                if plate_number:
                    result['success'] = True
                    result['plate_number'] = plate_number
                    result['confidence'] = 0.85
                    
                    # Draw rectangle
                    cv2.rectangle(result['image'], (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(result['image'], plate_number, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return result
            
        except Exception as e:
            logging.error(f"ANPR error: {e}")
            return result