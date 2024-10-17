#ifndef SD_MODEL_HPP
#define SD_MODEL_HPP

#include "stablediffusion.h"
#include "modelloader.h"

#include "core/object/ref_counted.h"
#include "core/object/class_db.h"

enum rng_type_t {
    STD_DEFAULT_RNG,
    CUDA_RNG
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
    
    Object *callback_receiver;
    StringName callback_method;

protected:
    static void _bind_methods();

public:
    int n_threads            = -1;
    bool free_params_immediately = false;
    bool print_log = false;
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
                        Object *receiver, const StringName &method);
    ~StableDiffusionGGML();

    /* Helper */
    void printlog(String out_log);

    bool is_using_v_parameterization_for_sd2(ggml_context* work_ctx);
    void calculate_alphas_cumprod(float* alphas_cumprod, float linear_start, float linear_end, int timesteps);

    /* Model */
    bool load_from_file(String str_model_path,
                        ggml_type wtype,

                        bool clip_on_cpu,String str_clip_l_path,String str_t5xxl_path,
                        bool control_net_cpu,String str_control_net_path,String str_embeddings_path,
                        String str_diffusion_model_path,String str_id_embeddings_path,schedule_t schedule,
                        bool vae_on_cpu,bool vae_only_decode,String str_vae_path,String str_taesd_path,

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
	SDVersion version;
	String model_path;

protected:
	static void _bind_methods();

public:
    Ref<StableDiffusionGGML> sd = nullptr;

    SDModel();
	~SDModel();

    void _on_sdmod_info(String info);

	void set_model_path(String p_path);
	String get_model_path() const;
	void set_backend(Backend p_backen);
	Backend get_backend() const;
	void set_version(SDVersion p_version);
	SDVersion get_version() const;
	void set_wtype(ggml_type p_wtype);
	ggml_type get_wtype() const;
};


#endif // SD_MODEL_HPP