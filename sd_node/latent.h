#ifndef LATENT_H
#define LATENT_H

#include "stablediffusion.h"
#include "ggml_extend.hpp"

class Latent : public SDResource {
	GDCLASS(Latent, SDResource);

private:
    int width = 512;
	int height = 512;
    int batch_count = 1;

    ggml_tensor* latent = NULL;
    struct ggml_context* work_ctx = NULL;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	void set_width(const int &p_width);
	int get_width() const;
    void set_height(const int &p_height);
	int get_height() const;

    void create_latent(SDVersion version);
    ggml_tensor* get_latent() const;
    struct ggml_context* get_work_ctx() const;

};

#endif // LATENT_H