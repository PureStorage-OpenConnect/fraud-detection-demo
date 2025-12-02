#!/usr/bin/env python3
"""
Pod 5: Notification Service
Handles fraud alerts from the inference service
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# In-memory storage for alerts (in production, use a database)
alerts = []

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'notification-service',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/notify/fraud', methods=['POST'])
def notify_fraud():
    """Receive fraud alert from inference service"""
    try:
        # Parse request data
        data = request.get_json()
        
        # Extract alert information
        alert = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': data.get('transaction_id'),
            'fraud_score': data.get('fraud_score'),
            'user_id': data.get('user_id'),
            'merchant_id': data.get('merchant_id'),
            'amount': data.get('amount'),
            'confidence': data.get('confidence', 0.0),
            'alert_type': 'HIGH_RISK' if data.get('fraud_score', 0) > 0.9 else 'MEDIUM_RISK'
        }
        
        # Store alert
        alerts.append(alert)
        
        # Log alert
        logger.warning(f"FRAUD ALERT: {alert['alert_type']} - "
                      f"Transaction {alert['transaction_id']} - "
                      f"Score: {alert['fraud_score']:.4f} - "
                      f"Amount: ${alert['amount']:.2f}")
        
        # In production, you would:
        # 1. Send email/SMS notifications
        # 2. Update fraud case management system
        # 3. Trigger automated blocking if high confidence
        # 4. Log to security information and event management (SIEM)
        
        return jsonify({
            'status': 'success',
            'message': 'Alert received and processed',
            'alert_id': len(alerts)
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing fraud alert: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Retrieve recent alerts"""
    limit = request.args.get('limit', default=100, type=int)
    risk_level = request.args.get('risk_level', default=None, type=str)
    
    # Filter alerts
    filtered_alerts = alerts
    if risk_level:
        filtered_alerts = [a for a in alerts if a['alert_type'] == risk_level]
    
    # Return most recent alerts
    recent_alerts = filtered_alerts[-limit:]
    
    return jsonify({
        'total_alerts': len(alerts),
        'filtered_count': len(filtered_alerts),
        'alerts': recent_alerts
    })

@app.route('/alerts/stats', methods=['GET'])
def get_alert_stats():
    """Get alert statistics"""
    if not alerts:
        return jsonify({
            'total_alerts': 0,
            'high_risk_count': 0,
            'medium_risk_count': 0
        })
    
    high_risk = sum(1 for a in alerts if a['alert_type'] == 'HIGH_RISK')
    medium_risk = sum(1 for a in alerts if a['alert_type'] == 'MEDIUM_RISK')
    
    # Calculate average fraud score
    avg_score = sum(a['fraud_score'] for a in alerts) / len(alerts)
    
    # Total amount at risk
    total_amount = sum(a['amount'] for a in alerts)
    
    return jsonify({
        'total_alerts': len(alerts),
        'high_risk_count': high_risk,
        'medium_risk_count': medium_risk,
        'avg_fraud_score': round(avg_score, 4),
        'total_amount_at_risk': round(total_amount, 2),
        'latest_alert': alerts[-1] if alerts else None
    })

@app.route('/alerts/clear', methods=['POST'])
def clear_alerts():
    """Clear all alerts (admin only)"""
    global alerts
    count = len(alerts)
    alerts = []
    logger.info(f"Cleared {count} alerts")
    
    return jsonify({
        'status': 'success',
        'message': f'Cleared {count} alerts'
    })

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Pod 5: Notification Service - Starting")
    logger.info("=" * 60)
    
    # Get configuration
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting notification service on {host}:{port}")
    
    # Run Flask app
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
