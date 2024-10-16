/*       By SleeeepyZhou        */

#ifndef LATENT_H
#define LATENT_H

#include "stablediffusion.h"
#include "ggml_extend.hpp"

/* Latent */
class Latent : public SDStableDiffusion {
	GDCLASS(Latent, StableDiffusion);

private:
    int width = 512;
	int height = 512;
    int batch_count = 1;

    struct ggml_context* work_ctx;
    struct ggml_tensor* latent = NULL;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	int get_width() const;
	int get_height() const;
	bool create_latent(SDVersion version);

    struct ggml_context* get_work_ctx() const;
    struct ggml_tensor* get_latent() const;
};

#endif // LATENT_H
