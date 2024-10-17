/*       By SleeeepyZhou        */

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"

/* loader */
class SDModelLoader : public StableDiffusion {
	GDCLASS(SDModelLoader, StableDiffusion);

private:
	

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();

// Helper

	// Node
	Backend create_backend(int device_index, bool use_cpu = false);
	Array load_model(Backend res_backend, 
					 String str_model_path, 
					 Scheduler schedule, 
					 bool clip_on_cpu = false, 
					 bool vae_on_cpu = false, 
					 bool vae_only_decode = false);
	/*
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
	*/
};

#endif // MODEL_LOADER_H