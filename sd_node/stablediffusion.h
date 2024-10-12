#ifndef STABLE_DIFFUSION_H
#define STABLE_DIFFUSION_H

#include "ai_object.h"

class SDResource : public AIResource {
	GDCLASS(SDResource, AIResource);

protected:
	static void _bind_methods();

public:
    SDResource();
    ~SDResource();
};

class StableDiffusion : public AIObject {
	GDCLASS(StableDiffusion, AIObject);
	bool print_log = false

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();
	
	void printlog(String p_log);

	void set_print_log(bool p_print_log);
	bool is_print_log() const;
};

#endif // STABLE_DIFFUSION_H
