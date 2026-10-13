/*
 * Multimodal Health Monitor Firmware for Arduino Uno R3
 * Sensors:
 *   - MAX30102 (PPG / SpO2 / Pulse Oximeter) over I2C (SDA=A4, SCL=A5)
 *   - AD8232 (Single-lead Heart Rate Monitor) over Analog A0, Leads-off on D10/D11
 *
 * Voltage Precautions:
 *   - AD8232 must be powered from the Arduino 3.3V rail (Max operating voltage: 3.6V).
 *   - MAX30102 must be powered from 3.3V (Sensor LEDs operate at 3.3V, IC core at 1.8V).
 *   - Connecting either sensor to 5.0V may permanently damage the sensor ICs.
 *
 * Serial Protocol:
 *   Baud rate: 115200
 *   Data Packet: SAMPLE,<time_ms>,<seq>,<ecg_val>,<ppg_red>,<ppg_ir>,<leads_off>\n
 *   Status Packet: STATUS,<code>,<message>\n
 */

#include <Wire.h>

// Pin definitions
const int PIN_ECG_ANALOG   = A0;
const int PIN_ECG_LO_PLUS  = 10;
const int PIN_ECG_LO_MINUS = 11;

// Sampling parameters
const unsigned long SAMPLE_INTERVAL_US = 10000; // 10,000 microseconds = 100 Hz
unsigned long last_sample_time_us = 0;
unsigned long sequence_number = 0;

// MAX30102 I2C Address and Registers
const uint8_t MAX30102_I2C_ADDR = 0x57;
const uint8_t REG_INTR_STATUS_1 = 0x00;
const uint8_t REG_INTR_ENABLE_1 = 0x02;
const uint8_t REG_FIFO_WR_PTR   = 0x04;
const uint8_t REG_OVF_COUNTER   = 0x05;
const uint8_t REG_FIFO_RD_PTR   = 0x06;
const uint8_t REG_FIFO_DATA     = 0x07;
const uint8_t REG_FIFO_CONFIG   = 0x08;
const uint8_t REG_MODE_CONFIG   = 0x09;
const uint8_t REG_SPO2_CONFIG   = 0x0A;
const uint8_t REG_LED1_PA       = 0x0C; // Red
const uint8_t REG_LED2_PA       = 0x0D; // IR
const uint8_t REG_PART_ID       = 0xFF;

bool max30102_present = false;
bool streaming_active = true;

// Helper to write to an I2C register
void i2c_write(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(MAX30102_I2C_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

// Helper to read an I2C register
uint8_t i2c_read(uint8_t reg) {
  Wire.beginTransmission(MAX30102_I2C_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(MAX30102_I2C_ADDR, (uint8_t)1);
  if (Wire.available()) {
    return Wire.read();
  }
  return 0;
}

// Initialize MAX30102
bool init_max30102() {
  Wire.begin();
  Wire.setClock(400000); // 400kHz Fast I2C mode

  // Test I2C communication by reading Part ID (0x15 for MAX30102)
  uint8_t part_id = i2c_read(REG_PART_ID);
  if (part_id != 0x15) {
    // If not matching default ID, probe device address presence
    Wire.beginTransmission(MAX30102_I2C_ADDR);
    if (Wire.endTransmission() != 0) {
      return false;
    }
  }

  // Reset MAX30102
  i2c_write(REG_MODE_CONFIG, 0x40);
  delay(10);

  // Clear FIFO pointers
  i2c_write(REG_FIFO_WR_PTR, 0x00);
  i2c_write(REG_OVF_COUNTER, 0x00);
  i2c_write(REG_FIFO_RD_PTR, 0x00);

  // FIFO Config: sample avg = 4, fifo rollover = enabled, fifo almost full = 17
  i2c_write(REG_FIFO_CONFIG, 0x4F);

  // Mode Config: SpO2 mode (Red & IR enabled)
  i2c_write(REG_MODE_CONFIG, 0x03);

  // SpO2 Config: ADC range = 4096nA, Sample Rate = 100 Hz, LED pulse width = 411us (18-bit resolution)
  i2c_write(REG_SPO2_CONFIG, 0x27);

  // LED Pulse Amplitude (current): Red = 0x24 (~7.2mA), IR = 0x24 (~7.2mA)
  i2c_write(REG_LED1_PA, 0x24);
  i2c_write(REG_LED2_PA, 0x24);

  return true;
}

// Read raw Red and IR samples from MAX30102 FIFO
bool read_max30102(uint32_t &red, uint32_t &ir) {
  if (!max30102_present) {
    red = 0;
    ir = 0;
    return false;
  }

  // Request 6 bytes (3 bytes Red, 3 bytes IR) from FIFO
  Wire.beginTransmission(MAX30102_I2C_ADDR);
  Wire.write(REG_FIFO_DATA);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  Wire.requestFrom(MAX30102_I2C_ADDR, (uint8_t)6);
  if (Wire.available() >= 6) {
    uint8_t r_b0 = Wire.read();
    uint8_t r_b1 = Wire.read();
    uint8_t r_b2 = Wire.read();
    uint8_t ir_b0 = Wire.read();
    uint8_t ir_b1 = Wire.read();
    uint8_t ir_b2 = Wire.read();

    red = (((uint32_t)r_b0 << 16) | ((uint32_t)r_b1 << 8) | (uint32_t)r_b2) & 0x03FFFF;
    ir  = (((uint32_t)ir_b0 << 16) | ((uint32_t)ir_b1 << 8) | (uint32_t)ir_b2) & 0x03FFFF;
    return true;
  }

  return false;
}

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  while (!Serial && millis() < 2000); // Wait for USB connection on Leonardo/Micro, continue on Uno

  // Configure AD8232 Pins
  pinMode(PIN_ECG_ANALOG, INPUT);
  pinMode(PIN_ECG_LO_PLUS, INPUT);
  pinMode(PIN_ECG_LO_MINUS, INPUT);

  // Initialize Sensors
  max30102_present = init_max30102();

  Serial.println(F("STATUS,READY,Multimodal Health Monitor v1.0"));
  if (max30102_present) {
    Serial.println(F("STATUS,MAX30102,OK"));
  } else {
    Serial.println(F("STATUS,MAX30102,NOT_DETECTED"));
  }
  Serial.println(F("STATUS,AD8232,ANALOG_A0_READY"));

  last_sample_time_us = micros();
}

void loop() {
  // Check for incoming serial commands
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd == "START") {
      streaming_active = true;
      Serial.println(F("STATUS,STREAM,STARTED"));
    } else if (cmd == "STOP") {
      streaming_active = false;
      Serial.println(F("STATUS,STREAM,STOPPED"));
    } else if (cmd == "PING") {
      Serial.println(F("STATUS,PONG,ALIVE"));
    } else if (cmd == "RESET") {
      sequence_number = 0;
      max30102_present = init_max30102();
      Serial.println(F("STATUS,RESET,COMPLETED"));
    }
  }

  if (!streaming_active) {
    delay(10);
    return;
  }

  // Non-blocking timer for 100 Hz sampling
  unsigned long current_us = micros();
  if (current_us - last_sample_time_us >= SAMPLE_INTERVAL_US) {
    last_sample_time_us += SAMPLE_INTERVAL_US;

    // 1. Read AD8232 ECG
    int ecg_raw = analogRead(PIN_ECG_ANALOG);
    int lo_plus = digitalRead(PIN_ECG_LO_PLUS);
    int lo_minus = digitalRead(PIN_ECG_LO_MINUS);
    int leads_off = (lo_plus == HIGH || lo_minus == HIGH) ? 1 : 0;

    // 2. Read MAX30102 PPG
    uint32_t red = 0;
    uint32_t ir = 0;
    read_max30102(red, ir);

    // 3. Emit formatted telemetry packet:
    // SAMPLE,<timestamp_ms>,<seq>,<ecg_val>,<ppg_red>,<ppg_ir>,<leads_off>
    Serial.print(F("SAMPLE,"));
    Serial.print(millis());
    Serial.print(',');
    Serial.print(sequence_number++);
    Serial.print(',');
    Serial.print(ecg_raw);
    Serial.print(',');
    Serial.print(red);
    Serial.print(',');
    Serial.print(ir);
    Serial.print(',');
    Serial.println(leads_off);
  }
}
