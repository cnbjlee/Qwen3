"""
Monitoring and logging system for Qwen3 deployments.

Features:
- Performance metrics collection
- Request/response logging
- Health monitoring
- Alert system
"""

import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import psutil
import requests


@dataclass
class RequestMetrics:
    """Request metrics data."""
    timestamp: float
    latency: float
    input_tokens: int
    output_tokens: int
    throughput: float
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System metrics data."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: List[float]
    gpu_memory: List[float]


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.request_metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_request(self, metrics: RequestMetrics) -> None:
        """Record request metrics."""
        with self._lock:
            self.request_metrics.append(metrics)
            self.counters['total_requests'] += 1
            
            if metrics.error:
                self.counters['error_requests'] += 1
            else:
                self.counters['success_requests'] += 1
                self.timers['latency'].append(metrics.latency)
                self.timers['throughput'].append(metrics.throughput)
    
    def record_system(self, metrics: SystemMetrics) -> None:
        """Record system metrics."""
        with self._lock:
            self.system_metrics.append(metrics)
    
    def get_stats(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get aggregated statistics."""
        with self._lock:
            now = time.time()
            window_start = now - (window_minutes * 60)
            
            # Filter recent metrics
            recent_requests = [
                m for m in self.request_metrics 
                if m.timestamp >= window_start and not m.error
            ]
            
            recent_system = [
                m for m in self.system_metrics
                if m.timestamp >= window_start
            ]
            
            # Calculate request stats
            if recent_requests:
                latencies = [m.latency for m in recent_requests]
                throughputs = [m.throughput for m in recent_requests]
                
                request_stats = {
                    'count': len(recent_requests),
                    'rps': len(recent_requests) / (window_minutes * 60),
                    'latency': {
                        'avg': sum(latencies) / len(latencies),
                        'p50': sorted(latencies)[len(latencies) // 2],
                        'p95': sorted(latencies)[int(len(latencies) * 0.95)],
                        'p99': sorted(latencies)[int(len(latencies) * 0.99)],
                    },
                    'throughput': {
                        'avg': sum(throughputs) / len(throughputs),
                        'total_tokens': sum(m.input_tokens + m.output_tokens for m in recent_requests),
                    }
                }
            else:
                request_stats = {'count': 0, 'rps': 0}
            
            # Calculate system stats
            if recent_system:
                system_stats = {
                    'cpu_avg': sum(m.cpu_usage for m in recent_system) / len(recent_system),
                    'memory_avg': sum(m.memory_usage for m in recent_system) / len(recent_system),
                    'gpu_avg': [
                        sum(m.gpu_usage[i] for m in recent_system if i < len(m.gpu_usage)) / 
                        len([m for m in recent_system if i < len(m.gpu_usage)])
                        for i in range(max(len(m.gpu_usage) for m in recent_system) if recent_system else 0)
                    ]
                }
            else:
                system_stats = {}
            
            return {
                'window_minutes': window_minutes,
                'timestamp': now,
                'requests': request_stats,
                'system': system_stats,
                'counters': dict(self.counters),
            }


class HealthMonitor:
    """Monitors service health."""
    
    def __init__(self, endpoint: str, check_interval: int = 30):
        self.endpoint = endpoint
        self.check_interval = check_interval
        self.is_healthy = False
        self.last_check = None
        self.failure_count = 0
        self._stop_event = threading.Event()
        self._thread = None
    
    def start(self) -> None:
        """Start health monitoring."""
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self) -> None:
        """Health monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                response = requests.get(f"{self.endpoint}/health", timeout=10)
                if response.status_code == 200:
                    self.is_healthy = True
                    self.failure_count = 0
                else:
                    self.is_healthy = False
                    self.failure_count += 1
            except requests.RequestException:
                self.is_healthy = False
                self.failure_count += 1
            
            self.last_check = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            'healthy': self.is_healthy,
            'last_check': self.last_check,
            'failure_count': self.failure_count,
        }


class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self, check_interval: int = 10):
        self.check_interval = check_interval
        self.metrics_collector = None
        self._stop_event = threading.Event()
        self._thread = None
    
    def start(self, metrics_collector: MetricsCollector) -> None:
        """Start system monitoring."""
        self.metrics_collector = metrics_collector
        
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self) -> None:
        """Stop system monitoring."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()
    
    def _monitor_loop(self) -> None:
        """System monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                # Get CPU and memory usage
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                
                # Get GPU usage (if available)
                gpu_usage = []
                gpu_memory = []
                
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    device_count = pynvml.nvmlDeviceGetCount()
                    
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        
                        # GPU utilization
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_usage.append(util.gpu)
                        
                        # GPU memory
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory.append((mem_info.used / mem_info.total) * 100)
                        
                except (ImportError, Exception):
                    # pynvml not available or error occurred
                    pass
                
                metrics = SystemMetrics(
                    timestamp=time.time(),
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_usage=gpu_usage,
                    gpu_memory=gpu_memory
                )
                
                if self.metrics_collector:
                    self.metrics_collector.record_system(metrics)
                    
            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")


class QwenLogger:
    """Custom logger for Qwen3."""
    
    def __init__(self, name: str = "qwen3", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('qwen3.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log_request(self, request_data: Dict[str, Any], response_data: Dict[str, Any], latency: float) -> None:
        """Log request/response data."""
        log_data = {
            'type': 'request',
            'timestamp': time.time(),
            'latency': latency,
            'request': request_data,
            'response': response_data,
        }
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with context."""
        log_data = {
            'type': 'error',
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'context': context or {},
        }
        self.logger.error(json.dumps(log_data, ensure_ascii=False))


class MonitoringDashboard:
    """Simple monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_monitor: HealthMonitor):
        self.metrics_collector = metrics_collector
        self.health_monitor = health_monitor
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        stats = self.metrics_collector.get_stats()
        health = self.health_monitor.get_status()
        
        return {
            'timestamp': time.time(),
            'health': health,
            'metrics': stats,
        }
    
    def print_dashboard(self) -> None:
        """Print dashboard to console."""
        data = self.get_dashboard_data()
        
        print("\n" + "="*60)
        print(f"🖥️  Qwen3 Monitoring Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Health status
        health = data['health']
        status_emoji = "✅" if health['healthy'] else "❌"
        print(f"\n🏥 Service Health: {status_emoji} {'Healthy' if health['healthy'] else 'Unhealthy'}")
        print(f"   Last Check: {datetime.fromtimestamp(health['last_check']).strftime('%H:%M:%S') if health['last_check'] else 'Never'}")
        print(f"   Failure Count: {health['failure_count']}")
        
        # Request metrics
        requests = data['metrics']['requests']
        if requests.get('count', 0) > 0:
            print(f"\n📊 Request Metrics (5min window):")
            print(f"   Total Requests: {requests['count']}")
            print(f"   Requests/sec: {requests['rps']:.2f}")
            print(f"   Avg Latency: {requests['latency']['avg']:.2f}ms")
            print(f"   P95 Latency: {requests['latency']['p95']:.2f}ms")
            print(f"   Avg Throughput: {requests['throughput']['avg']:.2f} tokens/s")
        else:
            print(f"\n📊 Request Metrics: No requests in the last 5 minutes")
        
        # System metrics
        system = data['metrics']['system']
        if system:
            print(f"\n💻 System Metrics:")
            print(f"   CPU Usage: {system.get('cpu_avg', 0):.1f}%")
            print(f"   Memory Usage: {system.get('memory_avg', 0):.1f}%")
            
            gpu_avg = system.get('gpu_avg', [])
            if gpu_avg:
                for i, usage in enumerate(gpu_avg):
                    print(f"   GPU {i} Usage: {usage:.1f}%")
        
        print("\n" + "="*60)