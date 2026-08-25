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

// Codec 2 초저대역폭 압축 라이브러리 (2400bps)
#include <codec2.h>

// Edge Impulse 모델 헤더
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 4
#include "dongjun1_inferencing.h"

#ifndef EI_CLASSIFIER_SLICE_SIZE
#define EI_CLASSIFIER_SLICE_SIZE (EI_CLASSIFIER_RAW_SAMPLE_COUNT / EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)
#endif

// ----------------- 1. 핀 및 통신 정의 -----------------
static constexpr int PIN_I2S_WS  = 7;
static constexpr int PIN_I2S_SCK = 6;
static constexpr int PIN_I2S_SD  = 5;
static constexpr i2s_port_t I2S_PORT = I2S_NUM_0;

#define RADAR1_RX_IO        17
#define RADAR1_TX_IO        18
#define RADAR2_RX_IO        15
#define RADAR2_TX_IO        16

#define LD2450_BAUDRATE     256000
#define LD2450_FRAME_LEN    30

HardwareSerial RadarSerial1(1);
HardwareSerial RadarSerial2(2);

// 비상 버튼 (외부 10kΩ 풀다운 저항, Active-HIGH)
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

// ----------------- 3. 전처리 및 필터 버퍼 -----------------
static constexpr float MIC_GAIN = 5.5f;
static constexpr float HPF_ALPHA = 0.95f;
static constexpr float NOISE_GATE_LEVEL = 70.0f;

static float s_filter_x = 0.0f;
static float s_filter_y = 0.0f;

static const float CONFIDENCE_THRESHOLD = 0.80f;
static const float MIC_LEVEL_THRESHOLD  = 0.04f;

static constexpr size_t MONO_SLICE_SIZE = EI_CLASSIFIER_SLICE_SIZE;
static constexpr size_t STEREO_BUFFER_SIZE = MONO_SLICE_SIZE * 2;
static int32_t rawStereoBuffer[STEREO_BUFFER_SIZE];
static int16_t g_audio_slice[MONO_SLICE_SIZE];
static size_t g_audio_slice_size = 0;

#define CALL_CHUNK_SAMPLES 960
static int32_t callRawBuffer[CALL_CHUNK_SAMPLES * 2];

static float g_last_microphone_level = 0.0f;
static uint16_t g_last_mic_avg = 0;

static uint32_t g_cooldown_until_ms = 0;
static int g_consecutive_hits = 0;
static int g_prev_detected_index = -1;

static volatile bool g_voice_call_active = false;
static bool g_prev_mode = false;

static struct CODEC2 *g_c2_enc = nullptr;

// ----------------- 4. BLE 콜백 -----------------
class ServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        g_ble_connected = true;
        Serial.println("🔗 [BLE] WROOM-32와 BLE 연결 성공!");
    }
    void onDisconnect(BLEServer* pServer) override {
        g_ble_connected = false;
        Serial.println("⚡ [BLE] 연결 해제됨 -> 광고(Advertising) 재개");
        BLEDevice::startAdvertising();
    }
};

class EmergencyCharCallbacks : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic *pChar) override {
        String rxValue = pChar->getValue().c_str();
        if (rxValue.length() > 0) {
            Serial.printf("📩 [BLE RX] 원격 제어 명령 수신: %s\n", rxValue.c_str());

            if (rxValue.indexOf("CALL_END") != -1) {
                g_voice_call_active = false;
                g_prev_mode = false;
                g_consecutive_hits = 0;
                g_prev_detected_index = -1;
                g_cooldown_until_ms = millis() + 4000U;
                i2s_zero_dma_buffer(I2S_PORT);

                Serial.println("📞 [SYSTEM] >>> 통화 종료: 대기 모드 100% 즉시 복귀!");
            } 
            else if (rxValue.indexOf("CALL_START") != -1) {
                g_voice_call_active = true;
                g_prev_mode = true;
                Serial.println("🚨 [SYSTEM] >>> 관제소 통화 연결 가동");
            }
        }
    }
};

static void init_ble() {
    BLEDevice::init("ESP32S3_EMERGENCY_SYSTEM");
    pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);

    // [핵심] PROPERTY_WRITE_NR (Write Without Response) 필수 활성화!
    pCharEmergency = pService->createCharacteristic(
        CHAR_EMERGENCY_UUID,
        BLECharacteristic::PROPERTY_READ | 
        BLECharacteristic::PROPERTY_NOTIFY | 
        BLECharacteristic::PROPERTY_WRITE | 
        BLECharacteristic::PROPERTY_WRITE_NR
    );
    pCharEmergency->setCallbacks(new EmergencyCharCallbacks());
    pCharEmergency->addDescriptor(new BLE2902());

    pCharRadar1 = pService->createCharacteristic(CHAR_RADAR1_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharRadar1->addDescriptor(new BLE2902());

    pCharRadar2 = pService->createCharacteristic(CHAR_RADAR2_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharRadar2->addDescriptor(new BLE2902());

    pCharAudioStream = pService->createCharacteristic(CHAR_AUDIO_STREAM_UUID, BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
    pCharAudioStream->addDescriptor(new BLE2902());

    pService->start();

    BLEAdvertising *pAdv = BLEDevice::getAdvertising();
    BLEAdvertisementData advData;
    advData.setFlags(0x06);
    advData.setCompleteServices(BLEUUID(SERVICE_UUID));
    pAdv->setAdvertisementData(advData);

    BLEAdvertisementData scanResponseData;
    scanResponseData.setName("ESP32S3_EMERGENCY_SYSTEM");
    pAdv->setScanResponseData(scanResponseData);

    pAdv->setMinPreferred(0x06);
    pAdv->start();
    Serial.println("📢 [BLE] GATT 서버 가동 완료 (Write_NR 활성화됨)");
}

static inline int16_t cleanProcessSample(int32_t rawLeft, int32_t rawRight) {
    int32_t raw = (abs(rawLeft) > abs(rawRight)) ? rawLeft : rawRight;
    int32_t s16 = raw >> 16;
    float x = static_cast<float>(s16);

    float y = HPF_ALPHA * (s_filter_y + x - s_filter_x);
    s_filter_x = x;
    s_filter_y = y;

    if (fabsf(y) < NOISE_GATE_LEVEL) {
        return 0;
    }

    float amplified = y * MIC_GAIN;
    if (amplified > 32000.0f) amplified = 32000.0f;
    if (amplified < -32000.0f) amplified = -32000.0f;

    return static_cast<int16_t>(amplified);
}

static bool init_i2s_microphone() {
    const i2s_config_t i2sConfig = {
        .mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
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
    return (strcmp(label, "saveme") == 0 || strcmp(label, "helpme") == 0 || strcmp(label, "goohe") == 0);
}

static const char *label_to_korean(const char *label) {
    if (strcmp(label, "saveme") == 0) return "살려주세요";
    if (strcmp(label, "helpme") == 0) return "도와주세요";
    if (strcmp(label, "goohe") == 0) return "구해주세요";
    return "";
}

static void process_inference_result(const ei_impulse_result_t &result, float mic_level) {
    if (millis() < g_cooldown_until_ms) {
        g_consecutive_hits = 0;
        g_prev_detected_index = -1;
        return;
    }

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
        g_cooldown_until_ms = millis() + 4000U;
        const char *korean_text = label_to_korean(best_label);

        Serial.printf("🚨 [AI 키워드 감지] %s (확신도: %.1f%%)\n", korean_text, best_confidence * 100.0f);

        if (g_ble_connected && pCharEmergency != nullptr) {
            pCharEmergency->setValue(korean_text);
            pCharEmergency->notify();
        }
        g_voice_call_active = true;
        g_prev_mode = true;
        g_consecutive_hits = 0;
        g_prev_detected_index = -1;
    }
}

// ----------------- 6. LD2450 초고속 레이더 파서 -----------------
static void process_radar_stream(HardwareSerial &port, BLECharacteristic *pChar, uint8_t &state_idx, uint8_t *tmp_buf) {
    while (port.available() > 0) {
        uint8_t byte = port.read();

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
        } else {
            tmp_buf[state_idx++] = byte;
            if (state_idx >= LD2450_FRAME_LEN) {
                if (tmp_buf[28] == 0x55 && tmp_buf[29] == 0xCC) {
                    if (g_ble_connected && pChar != nullptr && !g_voice_call_active) {
                        pChar->setValue(tmp_buf, LD2450_FRAME_LEN);
                        pChar->notify();
                    }
                }
                state_idx = 0;
            }
        }
    }
}

static void audio_classifier_task(void *arg) {
    (void)arg;

    while (true) {
        if (g_voice_call_active) {
            size_t bytesRead = 0;
            const size_t bytesToRead = CALL_CHUNK_SAMPLES * 2 * sizeof(int32_t);

            esp_err_t err = i2s_read(I2S_PORT, callRawBuffer, bytesToRead, &bytesRead, pdMS_TO_TICKS(50));
            if (g_voice_call_active && err == ESP_OK && bytesRead == bytesToRead && g_ble_connected && pCharAudioStream != nullptr && g_c2_enc != nullptr) {
                
                int16_t pcm8k[480];
                for (int i = 0; i < 480; i++) {
                    pcm8k[i] = cleanProcessSample(callRawBuffer[i * 4], callRawBuffer[i * 4 + 1]);
                }

                uint8_t c2_payload[18];
                codec2_encode(g_c2_enc, &c2_payload[0],  &pcm8k[0]);
                codec2_encode(g_c2_enc, &c2_payload[6],  &pcm8k[160]);
                codec2_encode(g_c2_enc, &c2_payload[12], &pcm8k[320]);

                pCharAudioStream->setValue(c2_payload, 18);
                pCharAudioStream->notify();
            }
            vTaskDelay(pdMS_TO_TICKS(1));
        } 
        else {
            size_t bytesRead = 0;
            const size_t bytesToRead = STEREO_BUFFER_SIZE * sizeof(int32_t);

            esp_err_t err = i2s_read(I2S_PORT, rawStereoBuffer, bytesToRead, &bytesRead, pdMS_TO_TICKS(100));
            if (err == ESP_OK && bytesRead > 0) {
                const size_t monoSamples = (bytesRead / sizeof(int32_t)) / 2;
                uint32_t levelSum = 0;

                for (size_t i = 0; i < monoSamples && i < MONO_SLICE_SIZE; ++i) {
                    int16_t sample = cleanProcessSample(rawStereoBuffer[i * 2], rawStereoBuffer[i * 2 + 1]);
                    g_audio_slice[i] = sample;
                    levelSum += abs(sample);
                }

                g_audio_slice_size = monoSamples;

                if (monoSamples > 0) {
                    g_last_mic_avg = static_cast<uint16_t>(levelSum / monoSamples);
                    g_last_microphone_level = static_cast<float>(g_last_mic_avg) / 10000.0f;
                    if (g_last_microphone_level > 1.0f) g_last_microphone_level = 1.0f;
                }

                signal_t signal;
                signal.total_length = g_audio_slice_size;
                signal.get_data = &raw_audio_signal_get_data;

                ei_impulse_result_t result = {0};
                EI_IMPULSE_ERROR ei_err = run_classifier_continuous(&signal, &result, false);

                if (ei_err == EI_IMPULSE_OK) {
                    process_inference_result(result, g_last_microphone_level);
                }
            }
            vTaskDelay(pdMS_TO_TICKS(5));
        }
    }
}

static void radar_task(void *arg) {
    (void)arg;
    uint8_t state1 = 0, state2 = 0;
    uint8_t tmp1[LD2450_FRAME_LEN], tmp2[LD2450_FRAME_LEN];

    while (true) {
        if (!g_voice_call_active) {
            process_radar_stream(RadarSerial1, pCharRadar1, state1, tmp1);
            process_radar_stream(RadarSerial2, pCharRadar2, state2, tmp2);
        } else {
            while (RadarSerial1.available()) RadarSerial1.read();
            while (RadarSerial2.available()) RadarSerial2.read();
            state1 = 0;
            state2 = 0;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

static void system_main_task(void *pvParameters) {
    Serial.println("🚀 [INIT] S3 시스템 초기화 워커 태스크 시작 (16KB 스택)");

    g_c2_enc = codec2_create(CODEC2_MODE_2400);

    RadarSerial1.setRxBufferSize(1024);
    RadarSerial1.begin(LD2450_BAUDRATE, SERIAL_8N1, RADAR1_RX_IO, RADAR1_TX_IO);

    RadarSerial2.setRxBufferSize(1024);
    RadarSerial2.begin(LD2450_BAUDRATE, SERIAL_8N1, RADAR2_RX_IO, RADAR2_TX_IO);

    pinMode(PIN_VOICE_CALL_BUTTON, INPUT);

    init_ble();

    if (!init_i2s_microphone()) {
        Serial.println("[FATAL] I2S 마이크 초기화 실패!");
        while (1) vTaskDelay(pdMS_TO_TICKS(1000));
    }

    xTaskCreatePinnedToCore(audio_classifier_task, "audioTask", 32768, nullptr, 2, nullptr, 1);
    xTaskCreatePinnedToCore(radar_task, "radarTask", 4096, nullptr, 1, nullptr, 0);

    Serial.println("\n[SYSTEM READY] ESP32-S3 고음질 Codec2 PTT 시스템 가동.\n");

    uint32_t btn_press_start_ms = 0;
    bool btn_is_pressed = false;
    bool btn_long_press_handled = false;

    while (true) {
        if (digitalRead(PIN_VOICE_CALL_BUTTON) == HIGH) {
            if (!btn_is_pressed) {
                btn_is_pressed = true;
                btn_press_start_ms = millis();
                btn_long_press_handled = false;
            } else if (!btn_long_press_handled) {
                if (millis() - btn_press_start_ms >= 3000U) {
                    g_voice_call_active = !g_voice_call_active;
                    btn_long_press_handled = true;
                    Serial.printf("🔘 [BUTTON] 3초 롱프레스 감지 -> 모드 변경: %s\n", g_voice_call_active ? "통화 모드" : "대기 모드");
                }
            }
        } else {
            btn_is_pressed = false;
            btn_long_press_handled = false;
        }

        if (g_prev_mode != g_voice_call_active) {
            g_prev_mode = g_voice_call_active;
            if (g_voice_call_active) {
                Serial.println("[MODE CHANGE] >>> 관제소 음성 대화 활성화");
                if (g_ble_connected && pCharEmergency != nullptr) {
                    pCharEmergency->setValue("CALL_START");
                    pCharEmergency->notify();
                }
            } else {
                Serial.println("[MODE CHANGE] >>> 일반 대기 모드 복귀");
                if (g_ble_connected && pCharEmergency != nullptr) {
                    pCharEmergency->setValue("CALL_END");
                    pCharEmergency->notify();
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
    xTaskCreatePinnedToCore(system_main_task, "sys_main_task", 16384, nullptr, 3, nullptr, 0);
}

void loop() {
    vTaskDelete(NULL);
}