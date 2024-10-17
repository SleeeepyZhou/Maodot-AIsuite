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

    struct ggml_context* work_ctx = NULL;
    struct ggml_tensor* latent = NULL;
    int latent_width;
	int latent_height;
    int latent_batch_count;
    float work_mem;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	void set_width(const int &p_width);
	int get_width() const;
	void set_height(const int &p_height);
	int get_height() const;
    void set_batch_count(const int &p_count);
    int get_batch_count() const;
	Array get_latent_info() const;

	void create_latent(SDVersion version = VERSION_SD1);

    struct ggml_context* get_work_ctx() const;
    struct ggml_tensor* get_latent() const;
    void free_work_ctx();
};

#endif // LATENT_H
