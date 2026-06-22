# - coding: utf-8 -
"""
Stability Monitor — VisionSpeak
================================
Provides crash logging, per-subsystem status tracking,
graceful degradation, and watchdog monitoring.

Usage:
    from system.stability_monitor import StabilityMonitor

    monitor = StabilityMonitor()
    monitor.start_watchdogs()

    # Wrap critical operations:
    with monitor.guard("camera"):
        frame = cap.read()
"""

import os
import sys
import json
import time
import traceback
import threading
from collections import defaultdict, deque
from datetime import datetime


CRASH_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs", "crash_log.txt"
)


# Ensure logs directory exists
os.makedirs(os.path.dirname(CRASH_LOG_PATH), exist_ok=True)


class SubsystemStatus:
    """Possible statuses for each subsystem."""
    RUNNING = "Running"
    WARNING = "Warning"
    FAILED = "Failed"
    DISABLED = "Disabled"


class StabilityMonitor:
    """
    Tracks the health of all critical subsystems, logs errors
    to a persistent crash log, and provides graceful degradation.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Per-subsystem status dictionary
        self._status = {
            "camera": SubsystemStatus.RUNNING,
            "mediapipe": SubsystemStatus.RUNNING,
            "prediction": SubsystemStatus.RUNNING,
            "emotion": SubsystemStatus.RUNNING,
            "supabase": SubsystemStatus.RUNNING,
            "tts": SubsystemStatus.RUNNING,
            "translation": SubsystemStatus.RUNNING,
            "display": SubsystemStatus.RUNNING,
        }

        # Error history for final report
        self._error_history = deque(maxlen=1000)  # (timestamp, subsystem, error_type, traceback)

        # Performance snapshot captured at crash time
        self._last_fps = 0.0
        self._last_frame_time_ms = 0.0
        self._camera_opened = False

        # Watchdog control
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_status(self, subsystem: str, status: str):
        """Update the health status of a subsystem."""
        with self._lock:
            self._status[subsystem] = status

    def get_status(self, subsystem: str) -> str:
        """Get the current health status of a subsystem."""
        with self._lock:
            return self._status.get(subsystem, SubsystemStatus.FAILED)

    def get_all_statuses(self) -> dict:
        """Return a copy of all subsystem statuses."""
        with self._lock:
            return dict(self._status)

    def report_error(self, subsystem: str, exc: BaseException, extra_context: dict = None):
        """
        Log an error to the crash log and update subsystem status.
        Call this whenever a subsystem raises an exception.
        """
        timestamp = datetime.now().isoformat()
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        entry = {
            "timestamp": timestamp,
            "subsystem": subsystem,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": tb_str,
            "fps": self._last_fps,
            "frame_time_ms": self._last_frame_time_ms,
            "camera_opened": self._camera_opened,
        }
        if extra_context:
            entry.update(extra_context)

        # Store in memory
        with self._lock:
            self._error_history.append(entry)
            self._status[subsystem] = SubsystemStatus.WARNING

        # Write to crash log
        self._write_crash_log(entry)

        # Print to console
        print(f"\n[STABILITY] ERROR in '{subsystem}': {type(exc).__name__}: {exc}")
        print(f"[STABILITY] See logs/crash_log.txt for full traceback.\n")

    def get_error_history(self) -> list:
        """Return a copy of the error history for the final report."""
        with self._lock:
            return list(self._error_history)

    # ── Context manager for guarded operations ──────────────────────────────

    def guard(self, subsystem: str, fallback_value=None, extra_context: dict = None):
        """
        Context manager that catches exceptions from a subsystem,
        logs them, and sets the subsystem status to Warning.
        Yields a dict so the caller can check if an error occurred.

        Usage:
            with monitor.guard("camera") as g:
                if not g["errored"]:
                    ret, frame = cap.read()
        """
        return _GuardContext(self, subsystem, fallback_value, extra_context)

    # ── Performance snapshot ────────────────────────────────────────────────

    def update_performance_snapshot(self, fps: float, frame_time_ms: float, camera_opened: bool):
        """Store the latest performance numbers for crash context."""
        self._last_fps = fps
        self._last_frame_time_ms = frame_time_ms
        self._camera_opened = camera_opened

    # ── Watchdog monitors ───────────────────────────────────────────────────

    def start_watchdogs(self, interval: float = 5.0):
        """
        Start a background watchdog thread that periodically checks
        the health of critical subsystems.
        """
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return  # already running
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(interval,),
            daemon=True,
            name="StabilityWatchdog",
        )
        self._watchdog_thread.start()
        print("[STABILITY] Watchdog monitors started.")

    def stop_watchdogs(self):
        """Signal the watchdog thread to stop."""
        self._watchdog_stop.set()
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)

    def _watchdog_loop(self, interval: float):
        """Background watchdog: checks subsystem health periodically."""
        while not self._watchdog_stop.is_set():
            with self._lock:
                for sub, status in self._status.items():
                    if status == SubsystemStatus.FAILED:
                        print(f"[WATCHDOG] Subsystem '{sub}' is FAILED.")
                    elif status == SubsystemStatus.WARNING:
                        pass  # already reported
            self._watchdog_stop.wait(interval)

    # ── Final report generation ─────────────────────────────────────────────

    def generate_final_report(self) -> str:
        """
        Build a text report summarising all errors encountered during
        the session. Printed at application exit.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  VisionSpeak — Stability Audit Final Report")
        lines.append("=" * 60)
        lines.append(f"  Generated: {datetime.now().isoformat()}")
        lines.append("")

        history = self.get_error_history()
        if not history:
            lines.append("  ✅ No errors recorded during this session.")
        else:
            # Count by subsystem
            subsystem_counts = defaultdict(int)
            error_type_counts = defaultdict(int)
            for entry in history:
                subsystem_counts[entry["subsystem"]] += 1
                error_type_counts[entry["error_type"]] += 1

            lines.append(f"  Total errors: {len(history)}")
            lines.append("")
            lines.append("  ── Failure Frequency by Subsystem ──")
            for sub, count in sorted(subsystem_counts.items(), key=lambda i: -i[1]):
                last_entry = [e for e in history if e["subsystem"] == sub][-1]
                lines.append(f"    {sub:20s}: {count} error(s)  (last: {last_entry['error_type']})")

            lines.append("")
            lines.append("  ── Error Types ──")
            for etype, count in sorted(error_type_counts.items(), key=lambda i: -i[1]):
                lines.append(f"    {etype:30s}: {count} occurrence(s)")

            lines.append("")
            lines.append("  ── Last 5 Errors (most recent first) ──")
            for entry in reversed(history[-5:]):
                lines.append(f"    [{entry['timestamp']}] {entry['subsystem']}: {entry['error_type']} — {entry['error_message'][:60]}")

            lines.append("")
            lines.append("  ── Recommended Actions ──")
            # Provide specific recommendations based on error patterns
            for sub in subsystem_counts:
                if sub == "camera":
                    lines.append(f"    * Camera: Check USB connection. Try using DirectShow backend: cv2.VideoCapture(0, cv2.CAP_DSHOW)")
                elif sub == "supabase":
                    lines.append(f"    * Supabase: Verify network connectivity and API keys. Consider retry logic with exponential backoff.")
                elif sub == "emotion":
                    lines.append(f"    * Emotion Detection: The FER/TensorFlow model may have crashed. Reinstalling the 'fer' library may help.")
                elif sub == "tts":
                    lines.append(f"    * Speech Engine: pyttsx3 may be missing a voice. Run 'python -m pyttsx3_info' to debug.")
                elif sub == "translation":
                    lines.append(f"    * Translation: googletrans might be rate-limited or offline. Check internet connection.")
                elif sub == "mediapipe":
                    lines.append(f"    * MediaPipe: Consider downscaling the processing resolution or updating the mediapipe package.")
                elif sub == "prediction":
                    lines.append(f"    * Prediction: The MLP model file may be corrupted. Re-run train_mlp.py to regenerate.")
                elif sub == "display":
                    lines.append(f"    * Display: The OpenCV window may have lost its context. Ensure only one cv2.imshow call per frame.")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _write_crash_log(self, entry: dict):
        """Append an error entry to the crash log file."""
        try:
            with open(CRASH_LOG_PATH, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"Timestamp: {entry['timestamp']}\n")
                f.write(f"Subsystem: {entry['subsystem']}\n")
                f.write(f"Error Type: {entry['error_type']}\n")
                f.write(f"Error Message: {entry['error_message']}\n")
                f.write(f"FPS at crash: {entry.get('fps', 'N/A')}\n")
                f.write(f"Frame time: {entry.get('frame_time_ms', 'N/A')} ms\n")
                f.write(f"Camera opened: {entry.get('camera_opened', 'N/A')}\n")
                f.write(f"Full Traceback:\n{entry['traceback']}\n")
                f.write("=" * 70 + "\n")
        except (IOError, OSError) as e:
            # If we cannot write to the crash log, print to stderr
            print(f"[STABILITY] CRITICAL: Failed to write crash log: {e}", file=sys.stderr)


class _GuardContext:
    """
    Context manager returned by StabilityMonitor.guard().
    Catches exceptions and reports them to the monitor.
    """

    def __init__(self, monitor: StabilityMonitor, subsystem: str,
                 fallback_value=None, extra_context: dict = None):
        self.monitor = monitor
        self.subsystem = subsystem
        self.extra_context = extra_context or {}
        self.result = {"errored": False, "exception": None, "value": fallback_value}

    def __enter__(self):
        return self.result

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return True  # no exception
        # Exception occurred — log and suppress
        self.result["errored"] = True
        self.result["exception"] = exc_val
        self.monitor.report_error(self.subsystem, exc_val, self.extra_context)
        # Prevent the exception from propagating
        return True
