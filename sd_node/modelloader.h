#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"
#include "ggml_extend.hpp"

#include "conditioner.hpp"
#include "diffusion_model.hpp"
#include "denoiser.hpp"


/* backend */
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

/* SDmodel -- Diffusion + CLIP */
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

	enum Schedule {
		DEFAULT,
		DISCRETE,
		KARRAS,
		EXPONENTIAL,
		AYS,
		GITS,
		N_SCHEDULES
	};

private:

	String model_path;
	SDVersion version;
	Schedule scheduler = DEFAULT;


	ggml_backend_t Backend				= NULL;  // general backend
    ggml_backend_t clip_backend			= NULL;
    ggml_type model_wtype				= GGML_TYPE_COUNT;
    ggml_type conditioner_wtype 		= GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype		= GGML_TYPE_COUNT;

    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<DiffusionModel> diffusion_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd
    std::shared_ptr<Denoiser> denoiser = std::make_shared<CompVisDenoiser>();

    float scale_factor       = 0.18215f;


    bool free_params_immediately = false;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();

    std::shared_ptr<DiffusionModel> diffusion_model;

    std::map<std::string, struct ggml_tensor*> tensors;


protected:
	static void _bind_methods();

public:
    SDModel();
    ~SDModel();
	String get_model_path() const;
	SDVersion get_version() const;
};

class CLIP : public SDResource {
	GDCLASS(SDModel, SDResource);

	ggml_backend_t clip_backend = NULL;
    ggml_type conditioner_wtype = GGML_TYPE_COUNT;

}

/* VAE TinyAE */
class VAEModel : public SDResource {
	GDCLASS(VAEModel, SDResource);

	String vae_path;

protected:
	static void _bind_methods();

public:
    VAEModel();
    ~VAEModel();
	void set_vae(const String &p_model_path);
	void _set_vae_path(const String &p_model_path);
	String _get_vae_path() const;

	void loading();
};

/* loader */
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
	};

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();
	
	Backend create_backend(int device_index, bool use_cpu = false);
	Array load_model(String model_path, 
					Backend backend, 
					Scheduler schedule = DEFAULT, 
					bool vae_only_decode = false,
					String clip_path = "",
					String t5xxl_path = "");
	VAEModel load_vae(String vae_path, bool only_decode = false);
	
};

#endif // MODEL_LOADER_H