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

// Initialize MAX30102 / MAX30100
bool is_max30100 = false;

bool init_max30102() {
  Wire.begin();
  Wire.setClock(100000); // 100kHz Standard I2C mode (reliable on jumper wires)

  // Step 1: Probe I2C address 0x57
  Wire.beginTransmission(MAX30102_I2C_ADDR);
  byte error = Wire.endTransmission();
  if (error != 0) {
    // Device not responding on 0x57
    return false;
  }

  // Step 2: Read Part ID (0x15 for MAX30102, 0x11 for MAX30100)
  uint8_t part_id = i2c_read(REG_PART_ID);

  if (part_id == 0x11) {
    // Module is MAX30100
    is_max30100 = true;
    i2c_write(0x06, 0x40); // Reset
    delay(50);
    i2c_write(0x06, 0x03); // SpO2 Mode
    i2c_write(0x07, 0x07); // 100 Hz, 1600us pulse
    i2c_write(0x09, 0xFF); // Full LED current (Red + IR)
    return true;
  }

  // Module is MAX30102 / MAX30105
  is_max30100 = false;

  // Reset MAX30102
  i2c_write(REG_MODE_CONFIG, 0x40);
  delay(50);

  // Clear FIFO pointers
  i2c_write(REG_FIFO_WR_PTR, 0x00);
  i2c_write(REG_OVF_COUNTER, 0x00);
  i2c_write(REG_FIFO_RD_PTR, 0x00);

  // FIFO Config: sample avg = 4, fifo rollover = enabled
  i2c_write(REG_FIFO_CONFIG, 0x4F);

  // Mode Config: SpO2 mode (Red & IR enabled)
  i2c_write(REG_MODE_CONFIG, 0x03);

  // SpO2 Config: ADC range = 4096nA, Sample Rate = 100 Hz, LED pulse width = 411us
  i2c_write(REG_SPO2_CONFIG, 0x27);

  // LED Pulse Amplitude: 0x3F (~12.5mA) for clearly visible bright red glow
  i2c_write(REG_LED1_PA, 0x3F);
  i2c_write(REG_LED2_PA, 0x3F);

  return true;
}

// Read raw Red and IR samples from FIFO
bool read_max30102(uint32_t &red, uint32_t &ir) {
  if (!max30102_present) {
    red = 0;
    ir = 0;
    return false;
  }

  if (is_max30100) {
    // MAX30100: 4 bytes from register 0x05 (2 bytes Red, 2 bytes IR)
    Wire.beginTransmission(MAX30102_I2C_ADDR);
    Wire.write(0x05);
    if (Wire.endTransmission(false) != 0) return false;

    Wire.requestFrom(MAX30102_I2C_ADDR, (uint8_t)4);
    if (Wire.available() >= 4) {
      uint8_t r_h = Wire.read();
      uint8_t r_l = Wire.read();
      uint8_t ir_h = Wire.read();
      uint8_t ir_l = Wire.read();
      red = ((uint32_t)r_h << 8) | (uint32_t)r_l;
      ir  = ((uint32_t)ir_h << 8) | (uint32_t)ir_l;
      return true;
    }
    return false;
  }

  // MAX30102: 6 bytes from register 0x07 (3 bytes Red, 3 bytes IR)
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
  while (!Serial && millis() < 2000);

  // Configure AD8232 Pins
  pinMode(PIN_ECG_ANALOG, INPUT);
  pinMode(PIN_ECG_LO_PLUS, INPUT);
  pinMode(PIN_ECG_LO_MINUS, INPUT);

  // Enable Arduino internal pullups on I2C pins A4/A5 (fixes 1.8V bus hang on Chinese MAX modules)
  pinMode(A4, INPUT_PULLUP);
  pinMode(A5, INPUT_PULLUP);

  // Initialize Sensors
  max30102_present = init_max30102();

  Serial.println(F("STATUS,READY,Multimodal Health Monitor v1.1"));
  if (max30102_present) {
    if (is_max30100) {
      Serial.println(F("STATUS,SENSOR,MAX30100_ONLINE_LED_ON"));
    } else {
      Serial.println(F("STATUS,SENSOR,MAX30102_ONLINE_LED_ON"));
    }
  } else {
    Serial.println(F("STATUS,SENSOR,MAX_NOT_FOUND_CHECK_A4_A5_VCC"));
    Serial.println(F("STATUS,DIAGNOSTIC,SCANNING_ALL_I2C_ADDRESSES..."));
    int nDevices = 0;
    for (byte address = 1; address < 127; address++) {
      Wire.beginTransmission(address);
      byte error = Wire.endTransmission();
      if (error == 0) {
        Serial.print(F("STATUS,FOUND_I2C_DEVICE_AT_ADDR,0x"));
        if (address < 16) Serial.print("0");
        Serial.println(address, HEX);
        nDevices++;
      }
    }
    if (nDevices == 0) {
      Serial.println(F("STATUS,RESULT,ZERO_I2C_DEVICES_FOUND_ON_BUS"));
      Serial.println(F("DIAGNOSTIC_TIP_1: Check if SDA is in ANALOG A4 and SCL is in ANALOG A5 (not D4/D5!)"));
      Serial.println(F("DIAGNOSTIC_TIP_2: Try moving VIN from 3.3V to 5V (many breakout boards require 5V)"));
      Serial.println(F("DIAGNOSTIC_TIP_3: Ensure GND wire has solid contact"));
    }
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
