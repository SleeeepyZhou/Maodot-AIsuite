/*       By SleeeepyZhou        */

#ifndef VAE_NODE_H
#define VAE_NODE_H

#include "core/io/image.h"

#include "stablediffusion.h"

#include "vae.hpp"
#include "tae.hpp"

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
}

#endif // VAE_NODE_H