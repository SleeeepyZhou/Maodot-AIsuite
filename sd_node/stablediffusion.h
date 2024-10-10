#include "ai_object.h"
#include "scene/main/node.h"

#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

class StableDiffusion : public AIObject {
	GDCLASS(StableDiffusion, AIObject);
	bool print_log = false

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();
	void set_print_log(bool p_print_log);
	bool is_print_log() const;
};

#endif // STABLE_DIFFUSION_H
