/*       By SleeeepyZhou        */

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

private:
	int n_threads = -1;

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();

	void set_n_threads(bool p_threads);
	int get_n_threads() const;

	Array get_vk_devices() const;
};

#endif // STABLE_DIFFUSION_H
