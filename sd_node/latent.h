#ifndef LATENT_H
#define LATENT_H

#include "ggml.h"

struct latent_image {
    ggml_tensor* tensor = NULL;
};

class Latent : public SDResource {
	GDCLASS(Latent, SDResource);

    Size2 size;
    StringName latent;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	void set_size(const Size2 &p_size);
	Size2 get_size() const;
    void set_latent(const StringName &p_latent);
	StringName get_latent() const;

};

#endif // LATENT_H