from flask import Flask, render_template, request, jsonify, Response
from models.database import db, VehicleEntry
from models.vehicle_classifier import VehicleClassifier
from utils.anpr import ANPRProcessor
from utils.video_processor import VideoProcessor
import os
import cv2
from datetime import datetime
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'vehicle_entries.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize extensions
db.init_app(app)

# Initialize components with error handling
classifier = None
anpr = None
video_processor = None

try:
    classifier = VehicleClassifier()
    logger.info("Vehicle classifier initialized")
except Exception as e:
    logger.error(f"Failed to load classifier: {e}")
    classifier = VehicleClassifier(mock_mode=True)

try:
    anpr = ANPRProcessor()
    logger.info("ANPR processor initialized")
except Exception as e:
    logger.error(f"Failed to initialize ANPR: {e}")
    anpr = ANPRProcessor(mock_mode=True)

try:
    video_processor = VideoProcessor()
    logger.info("Video processor initialized")
except Exception as e:
    logger.error(f"Failed to initialize video processor: {e}")
    video_processor = VideoProcessor(mock_mode=True)

# Create tables
with app.app_context():
    try:
        db.create_all()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Database error: {e}")
        logger.info("Continuing with in-memory database...")
        # Fallback to in-memory database
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        db.init_app(app)
        with app.app_context():
            db.create_all()

# Global video processing flag
is_processing = False
video_capture = None

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/anpr')
def anpr_page():
    """ANPR interface"""
    return render_template('anpr.html')

@app.route('/classifier')
def classifier_page():
    """Video classifier interface"""
    return render_template('classifier.html')

@app.route('/entries')
def entries_page():
    """View database entries"""
    return render_template('entries.html')

# ==================== ANPR ENDPOINTS ====================

@app.route('/api/detect_plate', methods=['POST'])
def detect_plate():
    """Detect number plate from uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        result = anpr.detect_plate(image)
        
        if result['success']:
            # Save result image
            result_filename = f"result_{filename}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            cv2.imwrite(result_path, result['image'])
            
            # Save to database
            entry = VehicleEntry(
                vehicle_number=result['plate_number'],
                vehicle_type='Unknown',
                entry_time=datetime.utcnow(),
                image_path=result_filename,
                status='parked',
                confidence_score=result.get('confidence', 0.85)
            )
            db.session.add(entry)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'plate_number': result['plate_number'],
                'result_image': result_filename,
                'entry_id': entry.id,
                'message': f'Plate detected: {result["plate_number"]}'
            })
        
        return jsonify({'success': False, 'message': 'No plate detected in image'})
        
    except Exception as e:
        logger.error(f"Plate detection error: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== VIDEO CLASSIFIER ENDPOINTS ====================

@app.route('/api/classify_video', methods=['POST'])
def classify_video():
    """Classify vehicle in uploaded video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save video
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process video
        result = video_processor.classify_video(filepath, classifier)
        
        # Save to database if plate detected
        if result.get('plate_number'):
            try:
                entry = VehicleEntry(
                    vehicle_number=result['plate_number'],
                    vehicle_type=result.get('vehicle_type', 'Unknown'),
                    entry_time=datetime.utcnow(),
                    status='parked',
                    confidence_score=result.get('confidence', 0.85)
                )
                db.session.add(entry)
                db.session.commit()
                result['entry_id'] = entry.id
            except Exception as e:
                logger.error(f"Database save error: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Video classification error: {e}")
        return jsonify({'error': str(e)}), 500

# ==================== DATABASE ENDPOINTS ====================

@app.route('/api/get_entries')
def get_entries():
    """Get all vehicle entries"""
    try:
        entries = VehicleEntry.query.order_by(VehicleEntry.entry_time.desc()).all()
        return jsonify([entry.to_dict() for entry in entries])
    except Exception as e:
        logger.error(f"Error fetching entries: {e}")
        return jsonify([])

@app.route('/api/get_parked')
def get_parked():
    """Get parked vehicles"""
    try:
        entries = VehicleEntry.query.filter_by(status='parked').order_by(VehicleEntry.entry_time.desc()).all()
        return jsonify([entry.to_dict() for entry in entries])
    except Exception as e:
        logger.error(f"Error fetching parked: {e}")
        return jsonify([])

@app.route('/api/exit_vehicle', methods=['POST'])
def exit_vehicle():
    """Mark vehicle as exited"""
    try:
        data = request.json
        entry = VehicleEntry.query.get(data.get('id'))
        if entry:
            entry.exit_time = datetime.utcnow()
            entry.status = 'exited'
            db.session.commit()
            return jsonify({'success': True, 'message': 'Vehicle marked as exited'})
        return jsonify({'success': False, 'message': 'Vehicle not found'}), 404
    except Exception as e:
        logger.error(f"Exit error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def search():
    """Search vehicles"""
    try:
        query = request.args.get('q', '').upper()
        entries = VehicleEntry.query.filter(
            VehicleEntry.vehicle_number.contains(query)
        ).order_by(VehicleEntry.entry_time.desc()).all()
        return jsonify([entry.to_dict() for entry in entries])
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify([])

@app.route('/api/stats')
def get_stats():
    """Get statistics"""
    try:
        total = VehicleEntry.query.count()
        parked = VehicleEntry.query.filter_by(status='parked').count()
        today = VehicleEntry.query.filter(
            db.func.date(VehicleEntry.entry_time) == datetime.utcnow().date()
        ).count()
        
        # Vehicle type distribution
        types = db.session.query(
            VehicleEntry.vehicle_type,
            db.func.count(VehicleEntry.id)
        ).filter(VehicleEntry.vehicle_type != 'Unknown').group_by(VehicleEntry.vehicle_type).all()
        
        return jsonify({
            'total': total,
            'parked': parked,
            'today': today,
            'types': [{'name': t[0], 'count': t[1]} for t in types]
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'total': 0, 'parked': 0, 'today': 0, 'types': []})

# ==================== VIDEO STREAMING ====================

@app.route('/video_feed')
def video_feed():
    """Real-time video feed with detection"""
    global is_processing, video_capture
    
    def generate():
        global is_processing, video_capture
        
        try:
            if video_capture is None:
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    logger.error("Could not open camera")
                    yield (b'--frame\r\n'
                           b'Content-Type: text/plain\r\n\r\nCamera not available\r\n')
                    return
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            is_processing = True
            frame_count = 0
            
            while is_processing:
                success, frame = video_capture.read()
                if not success:
                    break
                
                frame_count += 1
                
                # Process every 10th frame
                if frame_count % 10 == 0 and anpr:
                    # Detect plate
                    result = anpr.detect_plate(frame)
                    
                    if result['success']:
                        frame = result['image']
                        
                        # Save to database (avoid duplicates)
                        try:
                            # Check if same plate detected recently
                            recent = VehicleEntry.query.filter(
                                VehicleEntry.vehicle_number == result['plate_number'],
                                VehicleEntry.entry_time > datetime.utcnow().replace(hour=datetime.utcnow().hour-1)
                            ).first()
                            
                            if not recent:
                                entry = VehicleEntry(
                                    vehicle_number=result['plate_number'],
                                    entry_time=datetime.utcnow(),
                                    status='parked',
                                    confidence_score=0.85
                                )
                                db.session.add(entry)
                                db.session.commit()
                        except Exception as e:
                            logger.error(f"Database save error: {e}")
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        except Exception as e:
            logger.error(f"Video feed error: {e}")
        finally:
            if video_capture:
                video_capture.release()
                video_capture = None
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    """Stop video feed"""
    global is_processing
    is_processing = False
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    logger.info("Starting Vehicle Management System...")
    app.run(debug=True, host='0.0.0.0', port=5000)