from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class VehicleEntry(db.Model):
    """Vehicle entry model"""
    __tablename__ = 'vehicle_entries'
    
    id = db.Column(db.Integer, primary_key=True)
    vehicle_number = db.Column(db.String(50), nullable=False)
    vehicle_type = db.Column(db.String(50), default='Unknown')
    entry_time = db.Column(db.DateTime, default=datetime.utcnow)
    exit_time = db.Column(db.DateTime, nullable=True)
    image_path = db.Column(db.String(200), nullable=True)
    status = db.Column(db.String(20), default='parked')
    confidence_score = db.Column(db.Float, default=0.0)
    classification_score = db.Column(db.Float, default=0.0)
    camera_id = db.Column(db.String(20), default='Gate-1')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'vehicle_number': self.vehicle_number,
            'vehicle_type': self.vehicle_type,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M:%S') if self.entry_time else None,
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M:%S') if self.exit_time else None,
            'status': self.status,
            'confidence_score': f"{self.confidence_score:.2%}",
            'classification_score': f"{self.classification_score:.2%}",
            'duration': self.get_duration()
        }
    
    def get_duration(self):
        """Calculate parking duration"""
        if self.exit_time:
            duration = self.exit_time - self.entry_time
        else:
            duration = datetime.utcnow() - self.entry_time
        
        hours = duration.total_seconds() / 3600
        if hours < 1:
            minutes = int(duration.total_seconds() / 60)
            return f"{minutes} minutes"
        else:
            return f"{hours:.1f} hours"