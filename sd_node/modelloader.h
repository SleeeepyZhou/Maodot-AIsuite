/*       By SleeeepyZhou        */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"
#include "ggml_extend.hpp"

#include "conditioner.hpp"
#include "diffusion_model.hpp"
#include "denoiser.hpp"
#include "vae.hpp"
#include "tae.hpp"

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
	ggml_backend_t get_backend() const;

	Array get_vk_devices_idx() const;
	void set_device(int device_index);

	void usecpu(bool p_use);
	bool is_use_cpu() const;
}

/* SDmodel */
class SDModel : public SDResource {
	GDCLASS(SDModel, SDResource);
private:
	String model_path;

protected:
	static void _bind_methods();

public:
    SDModel();
	~SDModel();
	void set_model_path(String p_path);
	String get_model_path() const;
};

/* CLIP */
class CLIP : public SDModel {
	GDCLASS(CLIP, SDModel);

	Backend backend_res;
	String model_path;

	ggml_backend_t clip_backend = NULL;
    ggml_type conditioner_wtype = GGML_TYPE_COUNT;
	
    std::shared_ptr<Conditioner> cond_stage_model;
    std::shared_ptr<FrozenCLIPVisionEmbedder> clip_vision;  // for svd

}

/* Diffusion */
class Diffusion : public SDModel {
	GDCLASS(Diffusion, SDModel);

private:
	Backend backend_res;

	SDVersion version;
    float scale_factor       = 0.18215f;

	ggml_backend_t backend				= NULL;  // general backend
    ggml_type model_wtype				= GGML_TYPE_COUNT;
    ggml_type diffusion_model_wtype		= GGML_TYPE_COUNT;
    std::shared_ptr<DiffusionModel> diffusion_model;

	Scheduler schedule = DEFAULT;
    std::shared_ptr<Denoiser> denoiser;

    std::shared_ptr<RNG> rng = std::make_shared<STDDefaultRNG>();



protected:
	static void _bind_methods();

public:
    Diffusion();
	Diffusion(String model_path, 
			bool is_using_v_parameterization,
			Backend backend_res, 
			SDVersion version,
			Scheduler schedule);
	~Diffusion();
	SDVersion get_version() const;
};

/* VAE TinyAE */
class VAEModel : public SDModel {
	GDCLASS(VAEModel, SDModel);

protected:
	static void _bind_methods();

public:
    VAEModel();
    ~VAEModel();
};

/* loader */
class SDModelLoader : public StableDiffusion {
	GDCLASS(SDModelLoader, StableDiffusion);

private:
	const char* model_version_to_str[] = {
		"SD 1.x",
		"SD 2.x",
		"SDXL",
		"SVD",
		"SD3 2B",
		"Flux Dev",
		"Flux Schnell"};

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();

// Helper
	bool is_using_v_parameterization_for_sd2(ggml_context *work_ctx);
	void calculate_alphas_cumprod(float *alphas_cumprod, 
								float linear_start = 0.00085f, 
								float linear_end   = 0.0120, 
								int timesteps      = TIMESTEPS);
	Array load_from_file(Backend res_backend, 
						String str_model_path, 
						String str_clip_l_path, 
						String str_t5xxl_path, 
						String str_diffusion_model_path, 
						String str_vae_path, 
						String str_taesd_path, 
						ggml_type wtype			= GGML_TYPE_COUNT,
						schedule_t schedule		= DEFAULT, 
						bool clip_on_cpu        = false,
						bool vae_on_cpu         = false, 
						bool vae_only_decode    = false);

// Node
	Backend create_backend(int device_index, bool use_cpu = false);
	Array load_model(String model_path, Backend backend, 
					Scheduler schedule = DEFAULT, 
					bool vae_only_decode = false,
					bool clip_on_cpu = false,
					ggml_type wtype = GGML_TYPE_COUNT);
	CLIP load_clip(String model_path, Backend backend, 
					bool clip_on_cpu = false,
					String t5xxl_path = "",
					ggml_type wtype = GGML_TYPE_COUNT);
	SDModel load_diffusion(String model_path, Backend backend, 
							Scheduler schedule = DEFAULT,
							ggml_type wtype = GGML_TYPE_COUNT);
	VAEModel load_vae(String vae_path, Backend backend, 
						bool only_decode = false,
						bool use_tiny_ae = false,
						ggml_type wtype = GGML_TYPE_COUNT);
	
};

#endif // MODEL_LOADER_H