#include <Arduino.h>
#include <cmath>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/i2s.h"

// BLE 라이브러리
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ----------------- Edge Impulse 헤더 및 설정 -----------------
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4
#include "dongjun1_inferencing.h"

#ifndef EI_CLASSIFIER_SLICE_SIZE
#define EI_CLASSIFIER_SLICE_SIZE (EI_CLASSIFIER_RAW_SAMPLE_COUNT / EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)
#endif

// ----------------- 1. 핀 및 통신 정의 -----------------
// 마이크 (INMP441)
static constexpr int PIN_I2S_WS  = 7;
static constexpr int PIN_I2S_SCK = 6;
static constexpr int PIN_I2S_SD  = 5;
static constexpr i2s_port_t I2S_PORT = I2S_NUM_0;

// 레이더 1 (UART1)
#define RADAR1_RX_IO        17
#define RADAR1_TX_IO        18

// 레이더 2 (UART2)
#define RADAR2_RX_IO        15
#define RADAR2_TX_IO        16

#define LD2450_BAUDRATE     256000
#define LD2450_FRAME_LEN    30

HardwareSerial RadarSerial1(1);
HardwareSerial RadarSerial2(2);

// 비상 통화 토글 버튼 (Active LOW)
static constexpr gpio_num_t PIN_VOICE_CALL_BUTTON = GPIO_NUM_4;

// ----------------- 2. BLE UUID 정의 -----------------
#define SERVICE_UUID           "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHAR_EMERGENCY_UUID    "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CHAR_RADAR1_UUID       "1c95d5e3-d8f7-413a-bf3d-7a2e5d7be87e"
#define CHAR_RADAR2_UUID       "a22d2b58-e2eb-4d9b-819a-9e5c4a5d8e20"
#define CHAR_AUDIO_STREAM_UUID "d1a8b13e-6a54-4a2e-837a-123456789abc"

static BLEServer* pServer = nullptr;
static BLECharacteristic* pCharEmergency = nullptr;
static BLECharacteristic* pCharRadar1 = nullptr;
static BLECharacteristic* pCharRadar2 = nullptr;
static BLECharacteristic* pCharAudioStream = nullptr;
static bool g_ble_connected = false;

// ----------------- 3. 전처리 및 버퍼 정의 -----------------
static constexpr float MIC_GAIN = 2.0f;
static constexpr float HPF_ALPHA = 0.985f;
static float s_filter_x = 0.0f;
static float s_filter_y = 0.0f;

static const float CONFIDENCE_THRESHOLD = 0.80f;
static const float MIC_LEVEL_THRESHOLD  = 0.04f;

static constexpr size_t MONO_SLICE_SIZE = EI_CLASSIFIER_SLICE_SIZE;
static constexpr size_t STEREO_BUFFER_SIZE = MONO_SLICE_SIZE * 2;

static int32_t rawStereoBuffer[STEREO_BUFFER_SIZE];
static int16_t g_audio_slice[MONO_SLICE_SIZE];
static size_t g_audio_slice_size = 0;

static float g_last_microphone_level = 0.0f;
static uint16_t g_last_mic_avg = 0;
static uint32_t g_last_detection_ms = 0;
static int g_consecutive_hits = 0;
static int g_prev_detected_index = -1;

static volatile bool g_voice_call_active = false;

// 레이더 캐시 버퍼
static uint8_t g_radar1_latest_frame[LD2450_FRAME_LEN];
static bool g_radar1_has_new = false;
static uint8_t g_radar2_latest_frame[LD2450_FRAME_LEN];
static bool g_radar2_has_new = false;

// ----------------- 4. BLE 콜백 -----------------
class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        g_ble_connected = true;
        Serial.println("[BLE] WROOM 연결 완료");
    }
    void onDisconnect(BLEServer* pServer) override {
        g_ble_connected = false;
        Serial.println("[BLE] WROOM 연결 해제 -> 광고 재시작");
        BLEDevice::startAdvertising();
    }
};

static void init_ble() {
    BLEDevice::init("ESP32S3_EMERGENCY_SYSTEM");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);

    pCharEmergency = pService->createCharacteristic(
        CHAR_EMERGENCY_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharEmergency->addDescriptor(new BLE2902());

    pCharRadar1 = pService->createCharacteristic(
        CHAR_RADAR1_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharRadar1->addDescriptor(new BLE2902());

    pCharRadar2 = pService->createCharacteristic(
        CHAR_RADAR2_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharRadar2->addDescriptor(new BLE2902());

    pCharAudioStream = pService->createCharacteristic(
        CHAR_AUDIO_STREAM_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharAudioStream->addDescriptor(new BLE2902());

    pService->start();
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    pAdvertising->setMinPreferred(0x06);
    BLEDevice::startAdvertising();
    Serial.println("[BLE] 서버 활성화됨");
}

// ----------------- 5. 오디오 신호 전처리 -----------------
static inline int16_t processI2SSample(int32_t rawLeft) {
    int32_t sample16 = rawLeft >> 12;
    float x = static_cast<float>(sample16);
    float y = HPF_ALPHA * (s_filter_y + x - s_filter_x);
    s_filter_x = x;
    s_filter_y = y;

    float amplified = y * MIC_GAIN;
    if (amplified > 32767.0f) amplified = 32767.0f;
    if (amplified < -32768.0f) amplified = -32768.0f;

    return static_cast<int16_t>(amplified);
}

static bool init_i2s_microphone() {
    const i2s_config_t i2sConfig = {
        .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = (uint32_t)EI_CLASSIFIER_FREQUENCY,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 256,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0,
    };

    const i2s_pin_config_t pinConfig = {
        .bck_io_num = PIN_I2S_SCK,
        .ws_io_num = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = PIN_I2S_SD,
    };

    if (i2s_driver_install(I2S_PORT, &i2sConfig, 0, nullptr) != ESP_OK) return false;
    if (i2s_set_pin(I2S_PORT, &pinConfig) != ESP_OK) return false;
    i2s_zero_dma_buffer(I2S_PORT);
    return true;
}

static void capture_microphone_samples() {
    size_t bytesRead = 0;
    const size_t bytesToRead = STEREO_BUFFER_SIZE * sizeof(int32_t);

    esp_err_t err = i2s_read(I2S_PORT, rawStereoBuffer, bytesToRead, &bytesRead, pdMS_TO_TICKS(100));
    if (err != ESP_OK || bytesRead == 0) {
        g_audio_slice_size = 0;
        return;
    }

    const size_t monoSamples = (bytesRead / sizeof(int32_t)) / 2;
    uint32_t levelSum = 0;

    for (size_t i = 0; i < monoSamples && i < MONO_SLICE_SIZE; ++i) {
        int16_t sample = processI2SSample(rawStereoBuffer[i * 2]);
        g_audio_slice[i] = sample;
        levelSum += abs(sample);
    }

    g_audio_slice_size = monoSamples;

    if (monoSamples > 0) {
        g_last_mic_avg = static_cast<uint16_t>(levelSum / monoSamples);
        g_last_microphone_level = static_cast<float>(g_last_mic_avg) / 10000.0f;
        if (g_last_microphone_level > 1.0f) g_last_microphone_level = 1.0f;
    }
}

static int raw_audio_signal_get_data(size_t offset, size_t length, float *out_ptr) {
    if (offset + length > g_audio_slice_size) {
        for (size_t i = 0; i < length; ++i) out_ptr[i] = 0.0f;
        return 0;
    }
    for (size_t i = 0; i < length; ++i) {
        out_ptr[i] = static_cast<float>(g_audio_slice[offset + i]);
    }
    return 0;
}

static bool is_emergency_label(const char *label) {
    return (strcmp(label, "saveme") == 0 ||
            strcmp(label, "helpme") == 0 ||
            strcmp(label, "goohe") == 0);
}

static const char *label_to_korean(const char *label) {
    if (strcmp(label, "saveme") == 0) return "살려주세요";
    if (strcmp(label, "helpme") == 0) return "도와주세요";
    if (strcmp(label, "goohe") == 0) return "구해주세요";
    return "";
}

// ----------------- 6. 추론 결과 판별 및 BLE 전송 -----------------
static void process_inference_result(const ei_impulse_result_t &result, float mic_level) {
    if (mic_level < MIC_LEVEL_THRESHOLD) {
        g_consecutive_hits = 0;
        g_prev_detected_index = -1;
        return;
    }

    int best_index = -1;
    float best_confidence = 0.0f;

    for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; ++i) {
        float confidence = result.classification[i].value;
        if (confidence > best_confidence) {
            best_confidence = confidence;
            best_index = static_cast<int>(i);
        }
    }

    if (best_index == -1) return;

    const char *best_label = result.classification[best_index].label;

    if (!is_emergency_label(best_label) || best_confidence < CONFIDENCE_THRESHOLD) {
        g_consecutive_hits = 0;
        g_prev_detected_index = -1;
        return;
    }

    if (g_prev_detected_index == best_index) {
        g_consecutive_hits++;
    } else {
        g_prev_detected_index = best_index;
        g_consecutive_hits = 1;
    }

    if (g_consecutive_hits >= 2) {
        if (millis() - g_last_detection_ms >= 2000U) {
            g_last_detection_ms = millis();
            const char *korean_text = label_to_korean(best_label);

            Serial.printf("=> [%s] 감지됨 (확신도: %.1f%%)\n", korean_text, best_confidence * 100.0f);

            if (g_ble_connected && pCharEmergency != nullptr) {
                pCharEmergency->setValue(korean_text);
                pCharEmergency->notify();
            }
        }
        g_consecutive_hits = 0;
        g_prev_detected_index = -1;
    }
}

// ----------------- 7. LD2450 상태 머신 기반 슬라이딩 파서 -----------------
static void parse_radar_stream(HardwareSerial &port, uint8_t *dest_buf, bool &new_flag, uint8_t &state_idx, uint8_t *tmp_buf) {
    while (port.available() > 0) {
        uint8_t byte = port.read();

        // 1. 헤더 4바이트 동기화 (0xAA, 0xFF, 0x03, 0x00)
        if (state_idx == 0) {
            if (byte == 0xAA) { tmp_buf[0] = byte; state_idx = 1; }
        } else if (state_idx == 1) {
            if (byte == 0xFF) { tmp_buf[1] = byte; state_idx = 2; }
            else { state_idx = (byte == 0xAA) ? 1 : 0; }
        } else if (state_idx == 2) {
            if (byte == 0x03) { tmp_buf[2] = byte; state_idx = 3; }
            else { state_idx = 0; }
        } else if (state_idx == 3) {
            if (byte == 0x00) { tmp_buf[3] = byte; state_idx = 4; }
            else { state_idx = 0; }
        } 
        // 2. 바디 및 테일 수신 (4 ~ 29 바이트)
        else {
            tmp_buf[state_idx++] = byte;
            if (state_idx == LD2450_FRAME_LEN) {
                // 테일 검증 (0x55, 0xCC)
                if (tmp_buf[28] == 0x55 && tmp_buf[29] == 0xCC) {
                    memcpy(dest_buf, tmp_buf, LD2450_FRAME_LEN);
                    new_flag = true;
                }
                state_idx = 0; // 프레임 완료 후 리셋
            }
        }
    }
}

// ----------------- 8. FreeRTOS 태스크 -----------------
static void audio_classifier_task(void *arg) {
    (void)arg;

    while (true) {
        capture_microphone_samples();

        if (g_voice_call_active) {
            if (g_ble_connected && g_audio_slice_size > 0) {
                const uint8_t *bytePtr = reinterpret_cast<const uint8_t*>(g_audio_slice);
                size_t totalBytes = g_audio_slice_size * sizeof(int16_t);
                
                for (size_t offset = 0; offset < totalBytes; offset += 128) {
                    size_t chunk = (totalBytes - offset > 128) ? 128 : (totalBytes - offset);
                    pCharAudioStream->setValue(const_cast<uint8_t*>(bytePtr + offset), chunk);
                    pCharAudioStream->notify();
                }
            }
        } else {
            if (g_audio_slice_size > 0) {
                signal_t signal;
                signal.total_length = g_audio_slice_size;
                signal.get_data = &raw_audio_signal_get_data;

                ei_impulse_result_t result = {0};
                EI_IMPULSE_ERROR err = run_classifier_continuous(&signal, &result, false);

                if (err == EI_IMPULSE_OK) {
                    process_inference_result(result, g_last_microphone_level);
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

static void radar_task(void *arg) {
    (void)arg;
    uint8_t state1 = 0, state2 = 0;
    uint8_t tmp1[LD2450_FRAME_LEN], tmp2[LD2450_FRAME_LEN];
    uint32_t last_tx_ms = 0;

    while (true) {
        if (!g_voice_call_active) {
            // UART 스트림 상시 수신 및 프레임 완성
            parse_radar_stream(RadarSerial1, g_radar1_latest_frame, g_radar1_has_new, state1, tmp1);
            parse_radar_stream(RadarSerial2, g_radar2_latest_frame, g_radar2_has_new, state2, tmp2);

            // 약 10Hz(100ms) 고정 주기로 캐시된 최신 유효 프레임을 BLE Notify
            uint32_t now = millis();
            if (now - last_tx_ms >= 100) {
                last_tx_ms = now;

                if (g_ble_connected) {
                    if (g_radar1_has_new && pCharRadar1 != nullptr) {
                        pCharRadar1->setValue(g_radar1_latest_frame, LD2450_FRAME_LEN);
                        pCharRadar1->notify();
                        g_radar1_has_new = false;
                    }
                    if (g_radar2_has_new && pCharRadar2 != nullptr) {
                        pCharRadar2->setValue(g_radar2_latest_frame, LD2450_FRAME_LEN);
                        pCharRadar2->notify();
                        g_radar2_has_new = false;
                    }
                }
            }
        } else {
            while (RadarSerial1.available()) RadarSerial1.read();
            while (RadarSerial2.available()) RadarSerial2.read();
            state1 = 0;
            state2 = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}

// ----------------- 9. 버튼 인터럽트 핸들러 -----------------
static void IRAM_ATTR button_isr_handler() {
    static uint32_t last_press_ms = 0;
    uint32_t now = millis();
    if (now - last_press_ms > 300) {
        last_press_ms = now;
        g_voice_call_active = !g_voice_call_active;
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);

    // 1. 레이더 UART 설정
    RadarSerial1.setRxBufferSize(1024);
    RadarSerial1.begin(LD2450_BAUDRATE, SERIAL_8N1, RADAR1_RX_IO, RADAR1_TX_IO);

    RadarSerial2.setRxBufferSize(1024);
    RadarSerial2.begin(LD2450_BAUDRATE, SERIAL_8N1, RADAR2_RX_IO, RADAR2_TX_IO);

    // 2. 비상 버튼 초기화
    pinMode(PIN_VOICE_CALL_BUTTON, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_VOICE_CALL_BUTTON), button_isr_handler, FALLING);

    // 3. BLE 초기화
    init_ble();

    // 4. I2S 마이크 초기화
    if (!init_i2s_microphone()) {
        Serial.println("[FATAL] I2S 마이크 초기화 실패!");
        while (1) delay(1000);
    }

    // 5. 멀티코어 태스크 분배 (Core 1: AI/오디오, Core 0: 레이더)
    xTaskCreatePinnedToCore(audio_classifier_task, "audioTask", 32768, nullptr, 2, nullptr, 1);
    xTaskCreatePinnedToCore(radar_task, "radarTask", 4096, nullptr, 1, nullptr, 0);

    Serial.println("\n[SYSTEM READY] 시스템 가동 시작.");
}

void loop() {
    static bool prev_state = false;
    if (prev_state != g_voice_call_active) {
        prev_state = g_voice_call_active;
        if (g_voice_call_active) {
            Serial.println("[MODE CHANGE] >>> 관제소 음성 대화 모드 활성화 (AI/레이더 중단)");
            if (g_ble_connected) {
                pCharEmergency->setValue("CALL_START");
                pCharEmergency->notify();
            }
        } else {
            Serial.println("[MODE CHANGE] >>> 일반 대기 모드 복귀 (AI 추론 및 레이더 재개)");
            if (g_ble_connected) {
                pCharEmergency->setValue("CALL_END");
                pCharEmergency->notify();
            }
        }
    }
    delay(50);
}