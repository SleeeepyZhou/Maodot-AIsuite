#include "ai_object.h"
#include "scene/main/node.h"
#include "core/io/image.h"

#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

class StableDiffusion : public AIObject {
	GDCLASS(StableDiffusion, AIObject);

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();
	Ref<Image> t2i(String model_path, String prompt);
};

#endif // STABLE_DIFFUSION_H