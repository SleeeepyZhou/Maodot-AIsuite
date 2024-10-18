#ifndef __UTIL_H__
#define __UTIL_H__

#include <cstdint>
#include <string>
#include <vector>

// #include "stable-diffusion.h"

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG
};
enum SDVersion {
	VERSION_SD1,
	VERSION_SD2,
	VERSION_SDXL,
	VERSION_SVD,
	VERSION_SD3_2B,
	VERSION_FLUX_DEV,
	VERSION_FLUX_SCHNELL,
	VERSION_COUNT,
};
enum Scheduler {
	DEFAULT,
	DISCRETE,
	KARRAS,
	EXPONENTIAL,
	AYS,
	GITS,
	N_SCHEDULES
};
enum SamplerName {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    IPNDM,
    IPNDM_V,
    LCM,
    N_SAMPLE_METHODS
};
enum sd_log_level_t {
    SD_LOG_DEBUG,
    SD_LOG_INFO,
    SD_LOG_WARN,
    SD_LOG_ERROR
};

enum sd_type_t {
    SD_TYPE_F32  = 0,
    SD_TYPE_F16  = 1,
    SD_TYPE_Q4_0 = 2,
    SD_TYPE_Q4_1 = 3,
    // SD_TYPE_Q4_2 = 4, support has been removed
    // SD_TYPE_Q4_3 = 5, support has been removed
    SD_TYPE_Q5_0     = 6,
    SD_TYPE_Q5_1     = 7,
    SD_TYPE_Q8_0     = 8,
    SD_TYPE_Q8_1     = 9,
    SD_TYPE_Q2_K     = 10,
    SD_TYPE_Q3_K     = 11,
    SD_TYPE_Q4_K     = 12,
    SD_TYPE_Q5_K     = 13,
    SD_TYPE_Q6_K     = 14,
    SD_TYPE_Q8_K     = 15,
    SD_TYPE_IQ2_XXS  = 16,
    SD_TYPE_IQ2_XS   = 17,
    SD_TYPE_IQ3_XXS  = 18,
    SD_TYPE_IQ1_S    = 19,
    SD_TYPE_IQ4_NL   = 20,
    SD_TYPE_IQ3_S    = 21,
    SD_TYPE_IQ2_S    = 22,
    SD_TYPE_IQ4_XS   = 23,
    SD_TYPE_I8       = 24,
    SD_TYPE_I16      = 25,
    SD_TYPE_I32      = 26,
    SD_TYPE_I64      = 27,
    SD_TYPE_F64      = 28,
    SD_TYPE_IQ1_M    = 29,
    SD_TYPE_BF16     = 30,
    SD_TYPE_Q4_0_4_4 = 31,
    SD_TYPE_Q4_0_4_8 = 32,
    SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_COUNT,
};

const char* sd_type_name(enum sd_type_t type);

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data);

const char* model_version_to_str[] = {
    "SD 1.x",
    "SD 2.x",
    "SDXL",
    "SVD",
    "SD3 2B",
    "Flux Dev",
    "Flux Schnell"};

const char* sampling_methods_str[] = {
    "Euler A",
    "Euler",
    "Heun",
    "DPM2",
    "DPM++ (2s)",
    "DPM++ (2M)",
    "modified DPM++ (2M)",
    "iPNDM",
    "iPNDM_v",
    "LCM",
};

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

int32_t get_num_physical_cores();
const char* sd_get_system_info();
void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data);
void sd_set_progress_callback(sd_progress_cb_t cb, void* data);


bool ends_with(const std::string& str, const std::string& ending);
bool starts_with(const std::string& str, const std::string& start);
bool contains(const std::string& str, const std::string& substr);

std::string format(const char* fmt, ...);

void replace_all_chars(std::string& str, char target, char replacement);

bool file_exists(const std::string& filename);
bool is_directory(const std::string& path);
std::string get_full_path(const std::string& dir, const std::string& filename);

std::vector<std::string> get_files_from_dir(const std::string& dir);

std::u32string utf8_to_utf32(const std::string& utf8_str);
std::string utf32_to_utf8(const std::u32string& utf32_str);
std::u32string unicode_value_to_utf32(int unicode_value);

sd_image_t* preprocess_id_image(sd_image_t* img);

// std::string sd_basename(const std::string& path);

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    float* data;
} sd_image_f32_t;

void normalize_sd_image_f32_t(sd_image_f32_t image, float means[3], float stds[3]);

sd_image_f32_t sd_image_t_to_sd_image_f32_t(sd_image_t image);

sd_image_f32_t resize_sd_image_f32_t(sd_image_f32_t image, int target_width, int target_height);

sd_image_f32_t clip_preprocess(sd_image_f32_t image, int size);

std::string path_join(const std::string& p1, const std::string& p2);

void pretty_progress(int step, int steps, float time);

void log_printf(sd_log_level_t level, const char* file, int line, const char* format, ...);

std::string trim(const std::string& s);

std::vector<std::pair<std::string, float>> parse_prompt_attention(const std::string& text);

#define LOG_DEBUG(format, ...) log_printf(SD_LOG_DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) log_printf(SD_LOG_INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) log_printf(SD_LOG_WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
#define LOG_ERROR(format, ...) log_printf(SD_LOG_ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif  // __UTIL_H__
