/**
 * Fraud Detection Demo v2 - Dashboard JavaScript
 * Handles real-time updates, charts, and user interactions
 */

// Configuration
const API_BASE = '';
const POLL_INTERVAL = 100; // ms
const STAGES = ['ingest', 'data_prep', 'model_train', 'inference'];
const STAGE_NAMES = {
    'ingest': 'Ingest',
    'data_prep': 'Data Prep',
    'model_train': 'Model Train',
    'inference': 'Inference'
};
const STAGE_SUBTITLES = {
    'ingest': 'Reading transaction data from FlashBlade',
    'data_prep': 'Feature engineering and data transformation',
    'model_train': 'Training XGBoost fraud detection model',
    'inference': 'Scoring transactions and writing results to FlashBlade'
};

// State
let currentState = null;
let pollInterval = null;
let cpuChart = null;
let gpuChart = null;
let summaryChart = null;

// Help modal toggle
function toggleHelpModal() {
    const modal = document.getElementById('help-modal');
    modal.style.display = modal.style.display === 'none' ? 'flex' : 'none';
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const modal = document.getElementById('help-modal');
    if (e.target === modal) {
        modal.style.display = 'none';
    }
});

// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Format time as MM:SS.s
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`;
}

// Initialize ApexCharts for throughput
function initCharts() {
    const chartOptions = {
        chart: {
            type: 'area',
            height: 80,
            sparkline: { enabled: true },
            animations: {
                enabled: true,
                easing: 'linear',
                dynamicAnimation: { speed: 100 }
            },
            toolbar: { show: false }
        },
        stroke: { curve: 'smooth', width: 2 },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.4,
                opacityTo: 0.1
            }
        },
        tooltip: { enabled: false },
        xaxis: { type: 'numeric' },
        yaxis: { min: 0 }
    };

    // CPU Chart
    cpuChart = new ApexCharts(document.getElementById('cpu-chart'), {
        ...chartOptions,
        colors: ['#4A90D9'],
        series: [{ name: 'Throughput', data: [] }]
    });
    cpuChart.render();

    // GPU Chart
    gpuChart = new ApexCharts(document.getElementById('gpu-chart'), {
        ...chartOptions,
        colors: ['#00D4AA'],
        series: [{ name: 'Throughput', data: [] }]
    });
    gpuChart.render();
}

// Update stage navigation pills
function updateStageNav(state) {
    const pills = document.querySelectorAll('.stage-pill');
    pills.forEach((pill, idx) => {
        pill.classList.remove('active', 'completed');
        if (idx < state.current_stage_idx) {
            pill.classList.add('completed');
            pill.textContent = '✓ ' + STAGE_NAMES[STAGES[idx]];
        } else if (idx === state.current_stage_idx) {
            pill.classList.add('active');
            pill.textContent = STAGE_NAMES[STAGES[idx]];
        } else {
            pill.textContent = STAGE_NAMES[STAGES[idx]];
        }
    });
}

// Update worker metrics display
function updateWorkerMetrics(worker, metrics) {
    const prefix = worker;

    // Rows
    document.getElementById(`${prefix}-rows`).textContent = formatNumber(metrics.rows_processed);
    const pct = metrics.total_rows > 0 ? (metrics.rows_processed / metrics.total_rows * 100) : 0;
    document.getElementById(`${prefix}-progress`).style.width = `${pct}%`;
    document.getElementById(`${prefix}-progress-pct`).textContent = `${pct.toFixed(0)}%`;
    document.getElementById(`${prefix}-total-rows`).textContent = `of ${formatNumber(metrics.total_rows)}`;

    // Throughput (now MB/s)
    document.getElementById(`${prefix}-throughput`).textContent = metrics.throughput_mbps.toFixed(1);

    // Max Throughput
    document.getElementById(`${prefix}-max-throughput`).textContent = metrics.max_throughput_mbps.toFixed(1);

    // Timer
    document.getElementById(`${prefix}-timer`).textContent = formatTime(metrics.elapsed_seconds);

    // Status
    const statusEl = document.getElementById(`${prefix}-status`);
    if (metrics.is_complete) {
        statusEl.textContent = 'Completed';
        statusEl.className = 'path-status completed';
    } else if (metrics.rows_processed > 0 || metrics.is_training) {
        statusEl.textContent = metrics.is_training ? 'Training...' : 'Running';
        statusEl.className = metrics.is_training ? 'path-status running training' : 'path-status running';
    }

    // Update chart
    const chart = worker === 'cpu' ? cpuChart : gpuChart;
    if (chart && metrics.history && metrics.history.length > 0) {
        const data = metrics.history.map((h, i) => ({ x: i, y: h.throughput }));
        chart.updateSeries([{ data: data }]);
    }
}

// Update insight banner
function updateInsight(cpuMetrics, gpuMetrics) {
    const banner = document.getElementById('insight-banner');

    if (cpuMetrics.rows_processed > 0 || gpuMetrics.rows_processed > 0) {
        banner.style.display = 'flex';

        // Calculate current speedup
        if (gpuMetrics.elapsed_seconds > 0 && gpuMetrics.rows_processed > 0) {
            const cpuRate = cpuMetrics.rows_processed / Math.max(cpuMetrics.elapsed_seconds, 0.1);
            const gpuRate = gpuMetrics.rows_processed / gpuMetrics.elapsed_seconds;
            const speedup = gpuRate / Math.max(cpuRate, 1);

            document.getElementById('current-speedup').textContent = `${speedup.toFixed(1)}x`;
        }

        // Update insight text based on throughput difference
        const throughputRatio = gpuMetrics.throughput_mbps / Math.max(cpuMetrics.throughput_mbps, 0.1);
        if (throughputRatio > 2) {
            document.getElementById('insight-text').textContent =
                `GPU demanding ${throughputRatio.toFixed(1)}x higher throughput — FlashBlade delivering without bottleneck`;
        }
    } else {
        banner.style.display = 'none';
    }
}

// Update UI with current state
function updateUI(state) {
    currentState = state;

    // Update stage nav
    updateStageNav(state);

    // Show live indicator when running
    document.getElementById('live-indicator').style.display = state.is_running ? 'flex' : 'none';

    // Get current stage data
    const stageName = state.current_stage;
    if (stageName && state.stages[stageName]) {
        const stageData = state.stages[stageName];

        // Update stage info in control bar
        document.getElementById('stage-name-display').textContent = STAGE_NAMES[stageName];
        document.getElementById('stage-desc-display').textContent = STAGE_SUBTITLES[stageName];

        // Update worker metrics
        updateWorkerMetrics('cpu', stageData.cpu);
        updateWorkerMetrics('gpu', stageData.gpu);

        // Update insight
        updateInsight(stageData.cpu, stageData.gpu);

        // Update button state
        const startBtn = document.getElementById('btn-start');
        if (state.is_running) {
            startBtn.disabled = true;
            startBtn.textContent = 'Running...';
        } else if (stageData.cpu.is_complete && stageData.gpu.is_complete) {
            startBtn.disabled = false;
            if (state.current_stage_idx >= STAGES.length - 1) {
                startBtn.textContent = 'View Summary';
            } else {
                startBtn.textContent = 'Continue →';
            }
        } else {
            // Stage in progress but not marked running (edge case)
            startBtn.disabled = true;
            startBtn.textContent = 'Processing...';
        }
    } else if (state.current_stage_idx < 0) {
        // Not started yet
        document.getElementById('stage-name-display').textContent = 'Ready to Start';
        document.getElementById('stage-desc-display').textContent = 'Click Start to begin';
        document.getElementById('btn-start').disabled = false;
        document.getElementById('btn-start').textContent = 'Start Demo';
    }
}

// Poll for state updates
async function pollState() {
    try {
        const response = await fetch(`${API_BASE}/api/state`);
        const state = await response.json();
        updateUI(state);

        // Check if we should show summary
        if (state.current_stage_idx >= STAGES.length - 1 &&
            state.stages.inference &&
            state.stages.inference.cpu.is_complete &&
            state.stages.inference.gpu.is_complete &&
            !state.is_running) {
            // All stages complete, ready for summary
        }
    } catch (e) {
        console.error('Failed to poll state:', e);
    }
}

// Start the demo or continue to next stage
async function startDemo() {
    const btn = document.getElementById('btn-start');

    // Check if we should show summary
    if (currentState &&
        currentState.current_stage_idx >= STAGES.length - 1 &&
        currentState.stages.inference &&
        currentState.stages.inference.cpu.is_complete &&
        currentState.stages.inference.gpu.is_complete) {
        showSummary();
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Starting...';

    try {
        const response = await fetch(`${API_BASE}/api/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const result = await response.json();

        if (result.error) {
            if (result.show_summary) {
                showSummary();
            } else {
                alert(result.error);
                btn.disabled = false;
            }
        }
    } catch (e) {
        console.error('Failed to start:', e);
        btn.disabled = false;
        btn.textContent = 'Start Demo';
    }
}

// Reset the demo
async function resetDemo() {
    if (currentState && currentState.is_running) {
        alert('Cannot reset while demo is running');
        return;
    }

    try {
        await fetch(`${API_BASE}/api/reset`, { method: 'POST' });

        // Reset UI
        document.getElementById('stage-view').style.display = 'block';
        document.getElementById('summary-view').style.display = 'none';

        // Reset metrics displays
        ['cpu', 'gpu'].forEach(prefix => {
            document.getElementById(`${prefix}-rows`).textContent = '0';
            document.getElementById(`${prefix}-progress`).style.width = '0%';
            document.getElementById(`${prefix}-progress-pct`).textContent = '0%';
            document.getElementById(`${prefix}-total-rows`).textContent = 'of 0';
            document.getElementById(`${prefix}-throughput`).textContent = '0.0';
            document.getElementById(`${prefix}-max-throughput`).textContent = '0.0';
            document.getElementById(`${prefix}-timer`).textContent = '00:00.0';
            document.getElementById(`${prefix}-status`).textContent = 'Ready';
            document.getElementById(`${prefix}-status`).className = 'path-status';
        });

        // Reset stage info
        document.getElementById('stage-name-display').textContent = 'Ready to Start';
        document.getElementById('stage-desc-display').textContent = 'Click Start to begin';

        // Reset charts
        if (cpuChart) cpuChart.updateSeries([{ data: [] }]);
        if (gpuChart) gpuChart.updateSeries([{ data: [] }]);

        // Reset stage pills
        document.querySelectorAll('.stage-pill').forEach((pill, idx) => {
            pill.classList.remove('active', 'completed');
            pill.textContent = STAGE_NAMES[STAGES[idx]];
        });

        document.getElementById('insight-banner').style.display = 'none';
        document.getElementById('btn-start').textContent = 'Start Demo';
        document.getElementById('btn-start').disabled = false;

    } catch (e) {
        console.error('Failed to reset:', e);
    }
}

// Show summary view
async function showSummary() {
    try {
        const response = await fetch(`${API_BASE}/api/summary`);
        const summary = await response.json();

        // Hide stage view, show summary
        document.getElementById('stage-view').style.display = 'none';
        document.getElementById('summary-view').style.display = 'block';

        // Update overall speedup
        document.getElementById('overall-speedup').textContent =
            `${summary.totals.overall_speedup}x`;

        // Update totals
        document.getElementById('total-records').textContent =
            formatNumber(summary.totals.cpu_rows);
        document.getElementById('total-cpu-time').textContent =
            formatTime(summary.totals.cpu_time);
        document.getElementById('total-gpu-time').textContent =
            formatTime(summary.totals.gpu_time);

        // Build stage breakdown cards
        const breakdown = document.getElementById('stage-breakdown');
        breakdown.innerHTML = '';

        STAGES.forEach(stageName => {
            const stageData = summary.stages[stageName];
            if (stageData) {
                const card = document.createElement('div');
                card.className = 'stage-card';
                card.innerHTML = `
                    <div class="stage-name">${STAGE_NAMES[stageName]}</div>
                    <div class="stage-speedup">${stageData.speedup}x</div>
                    <div style="font-size: 13px; color: var(--text-secondary);">faster</div>
                    <div class="stage-times">
                        <div class="time-block">
                            <div class="time-label">CPU</div>
                            <div class="time-value cpu">${stageData.cpu_time.toFixed(1)}s</div>
                        </div>
                        <div class="time-block">
                            <div class="time-label">GPU</div>
                            <div class="time-value gpu">${stageData.gpu_time.toFixed(1)}s</div>
                        </div>
                    </div>
                `;
                breakdown.appendChild(card);
            }
        });

        // Build throughput comparison chart
        const chartData = {
            categories: STAGES.map(s => STAGE_NAMES[s]),
            cpuData: STAGES.map(s => summary.stages[s] ? summary.stages[s].cpu_throughput : 0),
            gpuData: STAGES.map(s => summary.stages[s] ? summary.stages[s].gpu_throughput : 0)
        };

        if (summaryChart) {
            summaryChart.destroy();
        }

        summaryChart = new ApexCharts(document.getElementById('summary-throughput-chart'), {
            chart: {
                type: 'bar',
                height: 250,
                toolbar: { show: false },
                background: 'transparent'
            },
            series: [
                { name: 'CPU', data: chartData.cpuData },
                { name: 'GPU', data: chartData.gpuData }
            ],
            colors: ['#4A90D9', '#00D4AA'],
            plotOptions: {
                bar: {
                    horizontal: false,
                    columnWidth: '60%',
                    borderRadius: 4
                }
            },
            dataLabels: {
                enabled: true,
                formatter: (val) => `${val.toFixed(1)}`,
                style: { fontSize: '11px', colors: ['#fff'] }
            },
            xaxis: {
                categories: chartData.categories,
                labels: { style: { colors: '#B0B0C0' } }
            },
            yaxis: {
                title: { text: 'Throughput (MB/s)', style: { color: '#B0B0C0' } },
                labels: { style: { colors: '#B0B0C0' } }
            },
            legend: {
                position: 'top',
                labels: { colors: '#B0B0C0' }
            },
            grid: {
                borderColor: '#3D3D5C'
            },
            theme: { mode: 'dark' }
        });
        summaryChart.render();

        // Update button
        document.getElementById('btn-start').textContent = 'Run Again';
        document.getElementById('btn-start').disabled = false;

    } catch (e) {
        console.error('Failed to show summary:', e);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    pollState();
    pollInterval = setInterval(pollState, POLL_INTERVAL);
});
