"""
Unit tests for Host Serial Manager and Hardware Simulator
"""

import time
import pytest
from hardware.serial_reader import SerialManager


@pytest.fixture
def serial_mgr():
    mgr = SerialManager(port=None, baudrate=115200, buffer_size=100, mock_mode=True)
    yield mgr
    mgr.stop()


def test_serial_packet_parsing():
    mgr = SerialManager(mock_mode=False)

    # Valid packet
    line = "SAMPLE,14250,55,512,65400,68200,0"
    parsed = mgr._parse_line(line)
    assert parsed is not None
    assert parsed == (14250, 55, 512, 65400, 68200, 0)

    # Malformed packet (wrong field count)
    bad_line = "SAMPLE,14250,55,512"
    assert mgr._parse_line(bad_line) is None

    # Corrupt value
    corrupt_line = "SAMPLE,14250,NaN,512,65400,68200,0"
    assert mgr._parse_line(corrupt_line) is None


def test_mock_hardware_stream(serial_mgr):
    serial_mgr.start()
    assert serial_mgr.connection_status == "SIMULATING"

    time.sleep(0.3)  # Accumulate ~30 samples at 100 Hz
    samples = serial_mgr.get_latest_samples(50)

    assert len(samples) >= 15
    first_sample = samples[0]
    # (timestamp_ms, seq, ecg_raw, ppg_red, ppg_ir, leads_off)
    assert len(first_sample) == 6
    assert 0 <= first_sample[2] <= 1023  # ECG ADC bounds
    assert first_sample[3] > 10000       # Red PPG count
    assert first_sample[4] > 10000       # IR PPG count


def test_dropped_packet_detection():
    mgr = SerialManager()
    assert mgr.dropped_packets == 0

    # Parse seq 10
    mgr._parse_line("SAMPLE,1000,10,512,60000,65000,0")
    assert mgr.last_seq == 10
    assert mgr.dropped_packets == 0

    # Parse seq 11 (continuous, no drop)
    mgr._parse_line("SAMPLE,1010,11,515,60010,65010,0")
    assert mgr.last_seq == 11
    assert mgr.dropped_packets == 0

    # Jump to seq 15 (lost 12, 13, 14 -> 3 dropped)
    mgr._parse_line("SAMPLE,1050,15,520,60020,65020,0")
    assert mgr.last_seq == 15
    assert mgr.dropped_packets == 3


def test_ring_buffer_bounded_queue():
    mgr = SerialManager(buffer_size=10)
    for i in range(25):
        mgr._parse_line(f"SAMPLE,{1000+i*10},{i},512,60000,65000,0")
        with mgr.lock:
            mgr.sample_buffer.append((1000 + i * 10, i, 512, 60000, 65000, 0))

    # Buffer should not exceed maxlen=10
    assert len(mgr.sample_buffer) == 10
    latest = mgr.get_latest_samples(5)
    assert len(latest) == 5
    assert latest[-1][1] == 24

    # pop_all_samples should drain the buffer
    drained = mgr.pop_all_samples()
    assert len(drained) == 10
    assert len(mgr.sample_buffer) == 0


def test_port_discovery():
    ports = SerialManager.list_available_ports()
    assert isinstance(ports, list)
    for p in ports:
        assert "device" in p
        assert "description" in p
        assert "hwid" in p


def test_auto_detect_arduino_port():
    detected = SerialManager.auto_detect_arduino_port()
    # On the physical machine with Arduino connected, it will be COM5, or a string/None
    if detected is not None:
        assert isinstance(detected, str)
        assert detected.startswith("COM") or detected.startswith("/dev/")


def test_simulator_controls(serial_mgr):
    # Test setting heart rate, leads off, and finger contact
    serial_mgr.set_simulator_controls(heart_rate=120.0, leads_off=1, finger_contact=False)
    assert serial_mgr._sim_heart_rate == 120.0
    assert serial_mgr._sim_leads_off == 1
    assert serial_mgr._sim_finger_contact is False

    serial_mgr.start()
    time.sleep(0.15)
    samples = serial_mgr.get_latest_samples(10)
    assert len(samples) > 0
    # Leads off should produce 1023 or 1015
    for s in samples:
        assert s[5] == 1
        assert s[2] in (1023, 1015)

