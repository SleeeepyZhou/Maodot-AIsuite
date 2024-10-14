#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"
#include "ggml_extend.hpp"

class Backend : public SDResource {
	GDCLASS(Backend, SDResource);

private:
	bool use_cpu = false;
	ggml_backend_t backend = NULL;  // general backend

protected:
	static void _bind_methods();

public:
	Backend();
	Backend(int _index, bool _cpu = false);
	~Backend();

	Array get_vk_devices_idx() const;
	void set_device(int device_index);

	void usecpu(bool p_use);
	bool is_use_cpu() const;
}

class SDModel : public SDResource {
	GDCLASS(SDModel, SDResource);

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

private:
    ggml_type model_wtype              = GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype    = GGML_TYPE_COUNT;

	String model_path;
	SDVersion version = VERSION_COUNT;


    
    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();
    int n_threads            = -1;
    float scale_factor       = 0.18215f;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<DiffusionModel> diffusion_model;

    std::map<std::string, struct ggml_tensor*> tensors;

    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

protected:
	static void _bind_methods();

public:
    SDModel();
    ~SDModel();
	String get_model_path() const;
	SDVersion get_version() const;
};

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

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();
	
	Backend create_backend(int device_index, bool use_cpu = false);
	Array load_model(String model_path, Scheduler schedule = DEFAULT);
};

#endif // MODEL_LOADER_H