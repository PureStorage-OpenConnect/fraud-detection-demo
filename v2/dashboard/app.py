#!/usr/bin/env python3
"""
Dashboard / Orchestrator for Fraud Detection Demo v2

Provides:
- Web UI with PureStorage branding
- Stage orchestration (start, pause, continue)
- Real-time metrics collection from workers
- Summary generation

Communicates with CPU and GPU workers via HTTP.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import requests
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Configuration
CPU_WORKER_URL = os.getenv('CPU_WORKER_URL', 'http://worker-cpu:5001')
GPU_WORKER_URL = os.getenv('GPU_WORKER_URL', 'http://worker-gpu:5002')
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '5000'))

app = Flask(__name__)
CORS(app)

# Stage definitions
STAGES = ['ingest', 'data_prep', 'model_train', 'inference']
STAGE_NAMES = {
    'ingest': 'Ingest',
    'data_prep': 'Data Prep',
    'model_train': 'Model Train',
    'inference': 'Inference'
}


@dataclass
class WorkerMetrics:
    """Metrics for a single worker."""
    rows_processed: int = 0
    total_rows: int = 0
    bytes_processed: int = 0
    throughput_mbps: float = 0.0
    max_throughput_mbps: float = 0.0
    elapsed_seconds: float = 0.0
    is_complete: bool = False
    is_training: bool = False  # True during model training
    error: Optional[str] = None
    history: list = field(default_factory=list)  # For throughput chart


@dataclass
class StageState:
    """State for a single stage."""
    cpu: WorkerMetrics = field(default_factory=WorkerMetrics)
    gpu: WorkerMetrics = field(default_factory=WorkerMetrics)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class DemoState:
    """Global demo state."""

    def __init__(self):
        self.current_stage_idx: int = -1  # -1 = not started
        self.stages: Dict[str, StageState] = {s: StageState() for s in STAGES}
        self.is_running: bool = False
        self.lock = threading.Lock()

    @property
    def current_stage(self) -> Optional[str]:
        if 0 <= self.current_stage_idx < len(STAGES):
            return STAGES[self.current_stage_idx]
        return None

    def reset(self):
        with self.lock:
            self.current_stage_idx = -1
            self.stages = {s: StageState() for s in STAGES}
            self.is_running = False

    def start_stage(self, stage: str):
        with self.lock:
            self.stages[stage].started_at = time.time()
            self.stages[stage].cpu = WorkerMetrics()
            self.stages[stage].gpu = WorkerMetrics()
            self.is_running = True

    def update_metrics(self, worker: str, stage: str, metrics: Dict[str, Any]):
        with self.lock:
            state = self.stages.get(stage)
            if not state:
                return

            worker_metrics = state.cpu if worker == 'cpu' else state.gpu
            worker_metrics.rows_processed = metrics.get('rows_processed', 0)
            worker_metrics.total_rows = metrics.get('total_rows', 0)
            worker_metrics.bytes_processed = metrics.get('bytes_processed', 0)
            worker_metrics.throughput_mbps = metrics.get('throughput_mbps', 0.0)
            worker_metrics.elapsed_seconds = metrics.get('elapsed_seconds', 0.0)
            worker_metrics.is_training = metrics.get('is_training', False)

            # Track max throughput
            if worker_metrics.throughput_mbps > worker_metrics.max_throughput_mbps:
                worker_metrics.max_throughput_mbps = worker_metrics.throughput_mbps

            # Add to history for chart
            worker_metrics.history.append({
                'time': time.time(),
                'throughput': metrics.get('throughput_mbps', 0.0)
            })
            # Keep last 100 points
            if len(worker_metrics.history) > 100:
                worker_metrics.history = worker_metrics.history[-100:]

    def complete_worker(self, worker: str, stage: str, metrics: Dict[str, Any]):
        with self.lock:
            state = self.stages.get(stage)
            if not state:
                return

            worker_metrics = state.cpu if worker == 'cpu' else state.gpu
            worker_metrics.rows_processed = metrics.get('rows_processed', 0)
            worker_metrics.total_rows = metrics.get('total_rows', 0)
            worker_metrics.bytes_processed = metrics.get('bytes_processed', 0)
            worker_metrics.throughput_mbps = metrics.get('throughput_mbps', 0.0)
            worker_metrics.elapsed_seconds = metrics.get('elapsed_seconds', 0.0)
            worker_metrics.is_complete = True
            worker_metrics.error = metrics.get('error')

            # Track max throughput on completion too
            if worker_metrics.throughput_mbps > worker_metrics.max_throughput_mbps:
                worker_metrics.max_throughput_mbps = worker_metrics.throughput_mbps

            # Check if both workers complete
            if state.cpu.is_complete and state.gpu.is_complete:
                state.completed_at = time.time()
                self.is_running = False

    def get_state_dict(self) -> Dict[str, Any]:
        """Get serializable state."""
        with self.lock:
            stages_dict = {}
            for name, state in self.stages.items():
                stages_dict[name] = {
                    'cpu': {
                        'rows_processed': state.cpu.rows_processed,
                        'total_rows': state.cpu.total_rows,
                        'bytes_processed': state.cpu.bytes_processed,
                        'throughput_mbps': state.cpu.throughput_mbps,
                        'max_throughput_mbps': state.cpu.max_throughput_mbps,
                        'elapsed_seconds': state.cpu.elapsed_seconds,
                        'is_complete': state.cpu.is_complete,
                        'is_training': state.cpu.is_training,
                        'error': state.cpu.error,
                        'history': state.cpu.history[-50:]  # Last 50 points
                    },
                    'gpu': {
                        'rows_processed': state.gpu.rows_processed,
                        'total_rows': state.gpu.total_rows,
                        'bytes_processed': state.gpu.bytes_processed,
                        'throughput_mbps': state.gpu.throughput_mbps,
                        'max_throughput_mbps': state.gpu.max_throughput_mbps,
                        'elapsed_seconds': state.gpu.elapsed_seconds,
                        'is_complete': state.gpu.is_complete,
                        'is_training': state.gpu.is_training,
                        'error': state.gpu.error,
                        'history': state.gpu.history[-50:]
                    },
                    'started_at': state.started_at,
                    'completed_at': state.completed_at
                }

            return {
                'current_stage_idx': self.current_stage_idx,
                'current_stage': self.current_stage,
                'is_running': self.is_running,
                'stages': stages_dict
            }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all stages."""
        with self.lock:
            summary = {
                'stages': {},
                'totals': {
                    'cpu_time': 0,
                    'gpu_time': 0,
                    'cpu_rows': 0,
                    'gpu_rows': 0
                }
            }

            for name, state in self.stages.items():
                if state.cpu.is_complete and state.gpu.is_complete:
                    cpu_time = state.cpu.elapsed_seconds
                    gpu_time = state.gpu.elapsed_seconds
                    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

                    summary['stages'][name] = {
                        'cpu_time': cpu_time,
                        'gpu_time': gpu_time,
                        'speedup': round(speedup, 2),
                        'cpu_throughput': state.cpu.max_throughput_mbps,
                        'gpu_throughput': state.gpu.max_throughput_mbps
                    }

                    summary['totals']['cpu_time'] += cpu_time
                    summary['totals']['gpu_time'] += gpu_time

                    # Only count rows from ingest stage (same records processed in all stages)
                    if name == 'ingest':
                        summary['totals']['cpu_rows'] = state.cpu.rows_processed
                        summary['totals']['gpu_rows'] = state.gpu.rows_processed

            total_cpu = summary['totals']['cpu_time']
            total_gpu = summary['totals']['gpu_time']
            summary['totals']['overall_speedup'] = round(
                total_cpu / total_gpu if total_gpu > 0 else 0, 2
            )

            return summary


# Global state
demo_state = DemoState()


def log(msg: str):
    """Print timestamped log message."""
    print(f"{datetime.now():%H:%M:%S} [Dashboard] {msg}", flush=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================
@app.route('/')
def index():
    """Serve main dashboard page."""
    return render_template('index.html')


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current demo state."""
    return jsonify(demo_state.get_state_dict())


@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get summary of completed stages."""
    return jsonify(demo_state.get_summary())


@app.route('/api/start', methods=['POST'])
def start_demo():
    """Start the demo from the beginning or continue to next stage."""
    data = request.json or {}

    if demo_state.is_running:
        return jsonify({'error': 'Demo is already running'}), 400

    # Determine which stage to run
    if demo_state.current_stage_idx < 0:
        # Starting fresh
        next_stage_idx = 0
    else:
        # Continue to next stage
        next_stage_idx = demo_state.current_stage_idx + 1

    if next_stage_idx >= len(STAGES):
        return jsonify({'error': 'All stages complete', 'show_summary': True}), 400

    stage = STAGES[next_stage_idx]
    demo_state.current_stage_idx = next_stage_idx
    demo_state.start_stage(stage)

    log(f"Starting stage: {stage}")

    # Start both workers
    errors = []
    for worker_name, url in [('CPU', CPU_WORKER_URL), ('GPU', GPU_WORKER_URL)]:
        try:
            response = requests.post(
                f"{url}/api/start",
                json={'stage': stage},
                timeout=5
            )
            if response.status_code != 200:
                errors.append(f"{worker_name}: {response.text}")
        except Exception as e:
            errors.append(f"{worker_name}: {str(e)}")

    if errors:
        log(f"Worker start errors: {errors}")

    return jsonify({
        'status': 'started',
        'stage': stage,
        'stage_name': STAGE_NAMES[stage],
        'errors': errors if errors else None
    })


@app.route('/api/reset', methods=['POST'])
def reset_demo():
    """Reset the demo to initial state."""
    if demo_state.is_running:
        return jsonify({'error': 'Cannot reset while running'}), 400

    demo_state.reset()

    # Reset workers
    for url in [CPU_WORKER_URL, GPU_WORKER_URL]:
        try:
            requests.post(f"{url}/api/reset", timeout=5)
        except:
            pass

    log("Demo reset")
    return jsonify({'status': 'reset'})


@app.route('/api/metrics', methods=['POST'])
def receive_metrics():
    """Receive metrics update from a worker."""
    data = request.json
    worker = data.get('worker')
    stage = data.get('stage')
    metrics = data.get('metrics', {})

    demo_state.update_metrics(worker, stage, metrics)
    return jsonify({'status': 'ok'})


@app.route('/api/complete', methods=['POST'])
def receive_complete():
    """Receive stage completion from a worker."""
    data = request.json
    worker = data.get('worker')
    stage = data.get('stage')
    metrics = data.get('metrics', {})

    log(f"Worker {worker} completed stage {stage}")
    demo_state.complete_worker(worker, stage, metrics)
    return jsonify({'status': 'ok'})


@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    # Check worker health
    workers = {}
    for name, url in [('cpu', CPU_WORKER_URL), ('gpu', GPU_WORKER_URL)]:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            workers[name] = response.status_code == 200
        except:
            workers[name] = False

    return jsonify({
        'status': 'healthy',
        'workers': workers
    })


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("Fraud Detection Demo v2 - Dashboard")
    log("=" * 60)
    log(f"CPU Worker: {CPU_WORKER_URL}")
    log(f"GPU Worker: {GPU_WORKER_URL}")
    log(f"Dashboard port: {DASHBOARD_PORT}")
    log("-" * 60)

    app.run(host='0.0.0.0', port=DASHBOARD_PORT, threaded=True)


if __name__ == '__main__':
    main()
