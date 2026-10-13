"""
Host-Side Serial Communication and Telemetry Manager
Supports real Arduino Uno R3 connection via PySerial, automatic reconnection,
bounded FIFO buffer protection, and a realistic synthetic hardware simulator.
"""

import math
import random
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import serial
import serial.tools.list_ports


class SerialManager:
    """
    Manages non-blocking serial acquisition from Arduino Uno R3.
    Includes thread-safe bounded buffers, error recovery, and hardware simulation mode.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        buffer_size: int = 2000,
        mock_mode: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.buffer_size = buffer_size
        self.mock_mode = mock_mode

        # Thread-safe ring buffer for telemetry samples
        # Each item: (timestamp_ms, seq, ecg_raw, ppg_red, ppg_ir, leads_off)
        self.sample_buffer: Deque[Tuple[int, int, int, int, int, int]] = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None

        self.connection_status = "DISCONNECTED"  # DISCONNECTED, CONNECTED, SIMULATING, ERROR
        self.last_error = ""
        self.dropped_packets = 0
        self.last_seq = -1

        # Simulator internal state
        self._sim_sample_idx = 0
        self._sim_heart_rate = 72.0
        self._sim_leads_off = 0
        self._sim_finger_contact = True

    @staticmethod
    def list_available_ports() -> List[Dict[str, str]]:
        """Scans and returns available system serial ports."""
        ports = serial.tools.list_ports.comports()
        result = []
        for p in ports:
            result.append(
                {
                    "device": p.device,
                    "description": p.description,
                    "hwid": p.hwid,
                }
            )
        return result

    def start(self):
        """Starts background data collection thread."""
        if self._running:
            return

        self._running = True
        if self.mock_mode:
            self.connection_status = "SIMULATING"
            self._thread = threading.Thread(target=self._mock_worker, daemon=True)
            self._thread.start()
        else:
            self._thread = threading.Thread(target=self._serial_worker, daemon=True)
            self._thread.start()

    def stop(self):
        """Stops background thread and releases serial resource."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None

        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        self.connection_status = "DISCONNECTED"

    def set_mock_mode(self, enabled: bool):
        """Toggles hardware simulation mode."""
        was_running = self._running
        if was_running:
            self.stop()
        self.mock_mode = enabled
        if was_running:
            self.start()

    def set_simulator_controls(self, heart_rate: float = 72.0, leads_off: int = 0, finger_contact: bool = True):
        """Allows testing UI and alert responses using synthetic controls."""
        self._sim_heart_rate = max(40.0, min(180.0, heart_rate))
        self._sim_leads_off = leads_off
        self._sim_finger_contact = finger_contact

    def get_latest_samples(self, n: int = 500) -> List[Tuple[int, int, int, int, int, int]]:
        """Retrieves up to n most recent samples from the ring buffer safely."""
        with self.lock:
            if not self.sample_buffer:
                return []
            slice_len = min(n, len(self.sample_buffer))
            # Return copy of recent items
            return list(self.sample_buffer)[-slice_len:]

    def pop_all_samples(self) -> List[Tuple[int, int, int, int, int, int]]:
        """Drains all accumulated samples from the buffer (used for database persistence)."""
        with self.lock:
            samples = list(self.sample_buffer)
            self.sample_buffer.clear()
            return samples

    def _parse_line(self, line: str) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        Parses serial telemetry line:
        SAMPLE,<time_ms>,<seq>,<ecg_raw>,<ppg_red>,<ppg_ir>,<leads_off>
        """
        line = line.strip()
        if not line.startswith("SAMPLE,"):
            return None

        parts = line.split(",")
        if len(parts) != 7:
            return None

        try:
            timestamp_ms = int(parts[1])
            seq = int(parts[2])
            ecg_raw = int(parts[3])
            ppg_red = int(parts[4])
            ppg_ir = int(parts[5])
            leads_off = int(parts[6])

            # Check sequence continuity
            if self.last_seq != -1 and seq > self.last_seq + 1:
                self.dropped_packets += (seq - self.last_seq - 1)
            self.last_seq = seq

            return (timestamp_ms, seq, ecg_raw, ppg_red, ppg_ir, leads_off)
        except ValueError:
            return None

    def _serial_worker(self):
        """Worker thread for physical serial connection."""
        while self._running:
            if not self.port:
                self.connection_status = "ERROR"
                self.last_error = "No serial port configured"
                time.sleep(1.0)
                continue

            try:
                self.connection_status = "CONNECTING"
                self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
                time.sleep(1.5)  # Allow Arduino DTR reset stabilization
                self.connection_status = "CONNECTED"
                self.last_error = ""

                # Handshake
                self._serial.write(b"START\n")

                while self._running and self._serial.is_open:
                    raw_line = self._serial.readline()
                    if not raw_line:
                        continue
                    try:
                        line = raw_line.decode("ascii", errors="replace")
                    except Exception:
                        continue

                    parsed = self._parse_line(line)
                    if parsed:
                        with self.lock:
                            self.sample_buffer.append(parsed)

            except serial.SerialException as e:
                self.connection_status = "ERROR"
                self.last_error = str(e)
                if self._serial:
                    try:
                        self._serial.close()
                    except Exception:
                        pass
                    self._serial = None
                time.sleep(2.0)  # Reconnect backoff
            except Exception as e:
                self.connection_status = "ERROR"
                self.last_error = str(e)
                time.sleep(1.0)

    def _mock_worker(self):
        """
        Generates realistic physiological 100 Hz waveforms for ECG and PPG.
        Simulates cardiac cycles, R-peaks, dicrotic notches, SpO2 absorption, and respiration.
        """
        fs = 100.0  # 100 Hz
        sample_period = 1.0 / fs
        next_sample_time = time.time()

        while self._running:
            now = time.time()
            if now < next_sample_time:
                time.sleep(max(0.001, next_sample_time - now))
                continue
            next_sample_time += sample_period

            t = self._sim_sample_idx * sample_period
            self._sim_sample_idx += 1
            timestamp_ms = int(time.time() * 1000)

            # Cardiac frequency with mild Respiratory Sinus Arrhythmia (0.2 Hz)
            resp_mod = 0.05 * math.sin(2 * math.pi * 0.25 * t)
            hr_instant = self._sim_heart_rate * (1.0 + resp_mod)
            cardiac_freq = hr_instant / 60.0
            phase = (t * cardiac_freq) % 1.0  # 0.0 to 1.0 cardiac cycle

            # 1. Synthetic ECG Waveform (AD8232 centered at 512, range 0-1023)
            if self._sim_leads_off == 1:
                # Disconnected lead produces saturated noise / flatline
                ecg_raw = 1023 if (self._sim_sample_idx % 2 == 0) else 1015
            else:
                baseline = 512.0 + 15.0 * math.sin(2 * math.pi * 0.25 * t)  # Baseline wander
                # P wave: phase ~0.15
                p_wave = 35.0 * math.exp(-((phase - 0.15) ** 2) / (2 * (0.025**2)))
                # Q wave: phase ~0.22
                q_wave = -40.0 * math.exp(-((phase - 0.22) ** 2) / (2 * (0.01**2)))
                # R wave (sharp peak): phase ~0.25
                r_wave = 380.0 * math.exp(-((phase - 0.25) ** 2) / (2 * (0.012**2)))
                # S wave: phase ~0.28
                s_wave = -80.0 * math.exp(-((phase - 0.28) ** 2) / (2 * (0.012**2)))
                # T wave: phase ~0.45
                t_wave = 70.0 * math.exp(-((phase - 0.45) ** 2) / (2 * (0.04**2)))
                noise = random.gauss(0, 3.0)

                ecg_raw = int(np.clip(baseline + p_wave + q_wave + r_wave + s_wave + t_wave + noise, 0, 1023))

            # 2. Synthetic MAX30102 PPG Waveform (Red & IR)
            # Pulse Arrival Time delay: PPG peak occurs ~220ms after ECG R-peak (phase + 0.22*cardiac_freq)
            ppg_phase = (phase - 0.22) % 1.0
            if not self._sim_finger_contact:
                # No finger: raw baseline values are low noise
                ppg_red = int(random.gauss(12000, 300))
                ppg_ir = int(random.gauss(15000, 300))
            else:
                # Normal pulsatile PPG with dicrotic notch
                # Primary systolic peak at ppg_phase ~0.15
                systolic = math.exp(-((ppg_phase - 0.15) ** 2) / (2 * (0.06**2)))
                # Dicrotic notch / diastolic peak at ppg_phase ~0.35
                diastolic = 0.4 * math.exp(-((ppg_phase - 0.35) ** 2) / (2 * (0.05**2)))
                pulse_waveform = systolic + diastolic

                dc_red = 62000.0
                dc_ir = 68000.0
                ac_amplitude = 3500.0

                # Modulation ratio R ~ 0.52 produces SpO2 ~ 97%
                red_val = dc_red + (ac_amplitude * 0.72) * pulse_waveform + random.gauss(0, 40)
                ir_val = dc_ir + (ac_amplitude * 1.0) * pulse_waveform + random.gauss(0, 40)

                ppg_red = int(np.clip(red_val, 0, 262143))
                ppg_ir = int(np.clip(ir_val, 0, 262143))

            with self.lock:
                self.sample_buffer.append(
                    (timestamp_ms, self._sim_sample_idx, ecg_raw, ppg_red, ppg_ir, self._sim_leads_off)
                )


