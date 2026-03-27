# 🚗 College Vehicle Management System

An integrated system for Automatic Number Plate Recognition (ANPR) and vehicle classification with real-time database tracking.

## Features

- **Automatic Number Plate Recognition** (ANPR)
  - Real-time camera feed with plate detection
  - Upload images for plate detection
  - OCR-based text extraction
  
- **Vehicle Classification**
  - Video upload and analysis
  - AI-powered vehicle type detection (Car, Bike, Truck, Bus)
  - Classification confidence scoring

- **Database Management**
  - SQLite database for vehicle entries
  - Timestamp tracking for entry/exit
  - Search and filter functionality
  - Export data to Excel

- **Web Dashboard**
  - Admin interface with statistics
  - Live camera feed
  - Vehicle records management
  - Responsive design for all devices

## Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLite with SQLAlchemy
- **OCR**: Tesseract
- **Computer Vision**: OpenCV
- **AI/ML**: Keras/TensorFlow
- **Frontend**: HTML5, CSS3, Bootstrap, JavaScript

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR
- Webcam (for live feed)

### Step 1: Clone the Repository

```bash
git clone https://github.com/hemanthkumar3000/vehicle-management-system.git
cd vehicle-management-system