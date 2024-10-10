#include "stablediffusion.h"
#include "scene/main/node.h"

#include "stable-diffusion.h"

#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

class ModelLoad : public StableDiffusion {
	GDCLASS(ModelLoad, StableDiffusion);

private:
    sd_ctx_t* SDModel

protected:
	static void _bind_methods();

public:
	ModelLoad();
	~ModelLoad();

    free_model();
};

#endif // MODEL_LOADER_H