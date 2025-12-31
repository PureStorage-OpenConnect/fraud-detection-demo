#!/usr/bin/env python3
"""
Pod 5: Notification Service
REST API for fraud alerts.
"""

import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

app = Flask(__name__)
alerts = []


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'notification',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/notify/fraud', methods=['POST'])
def notify_fraud():
    """Receive fraud alert."""
    try:
        data = request.get_json()
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'transaction_id': data.get('transaction_id'),
            'fraud_score': data.get('fraud_score'),
            'amount': data.get('amount'),
            'alert_type': 'HIGH_RISK' if data.get('fraud_score', 0) > 0.9 else 'MEDIUM_RISK'
        }
        
        alerts.append(alert)
        log.warning(f"FRAUD ALERT: {alert['alert_type']} | "
                   f"Score: {alert['fraud_score']:.4f} | "
                   f"Amount: ${alert['amount']:.2f}")
        
        return jsonify({'status': 'success', 'alert_id': len(alerts)}), 200
        
    except Exception as e:
        log.error(f"Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts."""
    limit = request.args.get('limit', default=100, type=int)
    return jsonify({
        'total': len(alerts),
        'alerts': alerts[-limit:]
    })


@app.route('/alerts/stats', methods=['GET'])
def get_stats():
    """Get alert statistics."""
    if not alerts:
        return jsonify({'total': 0, 'high_risk': 0, 'medium_risk': 0})
    
    high = sum(1 for a in alerts if a['alert_type'] == 'HIGH_RISK')
    return jsonify({
        'total': len(alerts),
        'high_risk': high,
        'medium_risk': len(alerts) - high,
        'avg_score': sum(a['fraud_score'] for a in alerts) / len(alerts)
    })


def main():
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    log.info(f"Starting notification service on {host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()