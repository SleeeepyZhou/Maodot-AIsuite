#ifndef VAE_NODE_H
#define VAE_NODE_H

#include "core/io/image.h"

#include "stablediffusion.h"
#include "modelloader.h"

#include "vae.hpp"
#include "tae.hpp"

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

class VAE : public StableDiffusion {
	GDCLASS(VAE, StableDiffusion);

	Ref<Image> input_image;

protected:
	static void _bind_methods();

public:
	VAE();
	~VAE();
    void set_in_image(const Ref<Image> &p_image);
	Ref<Image> get_in_image() const;
	
	Latent encode_image();
	Ref<Image> decode_latent();

	void load_model(String model_path);
    void free_model();
	void set_schedule(Scheduler p_schedule);
	Scheduler get_schedule() const;
}

#endif // VAE_NODE_H