#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

#include "ai_object.h"

#include "scene/main/node.h"
#include "scene/resources/texture.h"

class StableDiffusion : public AIObject {
	GDCLASS(StableDiffusion, AIObject);

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();
	Ref<Texture2D> t2i(String model_path, String prompt);
};

#endif // STABLE_DIFFUSION_H