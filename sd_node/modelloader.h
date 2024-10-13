#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"

class SDModel : public SDResource {
	GDCLASS(SDModel, SDResource);

	String model_path;

	ggml_backend_t backend             = NULL;  // general backend
    ggml_backend_t clip_backend        = NULL;
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type conditioner_wtype        = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;

    SDVersion version;
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

public:
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

protected:
	static void _bind_methods();

public:
    SDModel();
    ~SDModel();
    void set_model(const String &p_model_path);
	void _set_model_path(const String &p_model_path);
	String _get_model_path() const;
};

class CLIP : public SDResource {
	
}

class SDModelLoader : public StableDiffusion {
	GDCLASS(SDModelLoader, StableDiffusion);

public:
	enum Scheduler {
		DEFAULT,
		DISCRETE,
		KARRAS,
		EXPONENTIAL,
		AYS,
		GITS,
		N_SCHEDULES
	};

private:
    sd_ctx_t* SDModel;
	Scheduler schedule = DEFAULT;
	String lora_path;

	Dictionary model_list;

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();
	void set_schedule(Scheduler p_schedule);
	Scheduler get_schedule() const;
	
	void load_model(String model_path);
    void free_model();

};

#endif // MODEL_LOADER_H