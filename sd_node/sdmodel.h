/*    ModelLoader for Maodot    */
/*       By SleeeepyZhou        */

#ifndef SD_MODEL_H
#define SD_MODEL_H

#include "ggml_extend.hpp"

#include "stablediffusion.h"

#include "core/object/ref_counted.h"
#include "core/object/class_db.h"

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

/* StableDiffusionGGML */
class StableDiffusionGGML : public RefCounted {
    GDCLASS(SDMod, RefCounted);

private:
    const char* model_version_to_str[] = {
		"SD 1.x",
		"SD 2.x",
		"SDXL",
		"SVD",
		"SD3 2B",
		"Flux Dev",
		"Flux Schnell"};
    
    SDModel *receiver;
    StringName method;

protected:
    static void _bind_methods();

public:
    int n_threads            = -1;
    bool free_params_immediately = false;
    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    std::map<std::string, struct ggml_tensor*> tensors;

    /* CLIP */
    ggml_backend_t clip_backend        = NULL;
    ggml_type conditioner_wtype        = GGML_TYPE_COUNT;
    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd

    /* Controlnet */
    ggml_backend_t control_net_backend = NULL;
    std::shared_ptr<ControlNet> control_net;

    /* Diffusion */
    SDVersion version;
    ggml_backend_t backend             = NULL;  // general backend
    float scale_factor                 = 0.18215f;
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();
    
    /* Photo Maker */
    bool stacked_id           = false;
    std::shared_ptr<LoraModel> pmid_lora;
    std::shared_ptr<PhotoMakerIDEncoder> pmid_model;

    /* Lora */
    std::string lora_model_dir;
    std::unordered_map<std::string, float> curr_lora_state;

    /* VAE */
    ggml_backend_t vae_backend         = NULL;
    ggml_type vae_wtype                = GGML_TYPE_COUNT;
    bool vae_decode_only      = false;
    std::shared_ptr<AutoEncoderKL> first_stage_model;
    bool vae_tiling           = false;
    bool use_tiny_autoencoder = false;
    std::shared_ptr<TinyAutoEncoder> tae_first_stage;

    StableDiffusionGGML(int n_threads, rng_type_t rng_type, 
                        ggml_backend_t backend,
                        SDModel *receiver, const StringName &method);
    ~StableDiffusionGGML();

    /* Helper */
    void printlog(String out_log);

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx);
    void calculate_alphas_cumprod(float* alphas_cumprod, float linear_start, float linear_end, int timesteps);

    /* Model */
    bool load_from_file(String str_model_path,
                        ggml_type wtype,

                        bool clip_on_cpu, String str_clip_l_path, String str_t5xxl_path,
                        bool control_net_cpu, String str_control_net_path, String str_embeddings_path,
                        String str_diffusion_model_path, String str_id_embeddings_path, Scheduler schedule,
                        bool vae_on_cpu, bool vae_only_decode, String str_vae_path, String str_taesd_path,

                        bool vae_tiling_);
    void apply_lora(const std::string& lora_name, float multiplier);
    void apply_loras(const std::unordered_map<std::string, float>& lora_state);

    ggml_tensor* id_encoder(ggml_context* work_ctx,
                            ggml_tensor* init_img,
                            ggml_tensor* prompts_embeds,
                            std::vector<bool>& class_tokens_mask);
    SDCondition get_svd_condition(ggml_context* work_ctx,
                                  sd_image_t init_image,
                                  int width,
                                  int height,
                                  int fps                    = 6,
                                  int motion_bucket_id       = 127,
                                  float augmentation_level   = 0.f,
                                  bool force_zero_embeddings = false);

    ggml_tensor* sample(ggml_context* work_ctx,
                        ggml_tensor* init_latent,
                        ggml_tensor* noise,
                        SDCondition cond,
                        SDCondition uncond,
                        ggml_tensor* control_hint,
                        float control_strength,
                        float min_cfg,
                        float cfg_scale,
                        float guidance,
                        sample_method_t method,
                        const std::vector<float>& sigmas,
                        int start_merge_step,
                        SDCondition id_cond);
    // ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding
    ggml_tensor* get_first_stage_encoding(ggml_context* work_ctx, ggml_tensor* moments);
    ggml_tensor* compute_first_stage(ggml_context* work_ctx, ggml_tensor* x, bool decode);
    ggml_tensor* encode_first_stage(ggml_context* work_ctx, ggml_tensor* x);
    ggml_tensor* decode_first_stage(ggml_context* work_ctx, ggml_tensor* x);
};

/* SDmodel */
class SDModel : public StableDiffusion {
	GDCLASS(SDModel, StableDiffusion);
private:
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
        "LCM"};
	SDVersion version;
	String model_path;
    Scheduler schedule;

protected:
	static void _bind_methods();

public:
    Ref<StableDiffusionGGML> sd = nullptr;

    SDModel();
	~SDModel();

    /* Helper */
	Array get_vk_devices_idx() const;
	ggml_backend_t set_device(int device_index = -1, bool use_cpu = false);

    String get_model_path() const;
	SDVersion get_version() const;
    Scheduler get_schedule() const;

	/* Model */
	void load_model(String str_model_path, int device_index = -1, Scheduler schedule = DEFAULT,
                    bool use_cpu = false, bool vae_on_cpu = false, bool clip_on_cpu = false);

    /* Inference */
    void ksample(Latent init_latent, SDCond cond,
                 int steps = 10,
                 float CFG = 8.0f,
                 float denoise = 1.0f,
                 SamplerName sampler_name = LCM,
                 int seed = 42);

    /* VAE */
    void decode(Latent init_latent);
};


#endif // SD_MODEL_H