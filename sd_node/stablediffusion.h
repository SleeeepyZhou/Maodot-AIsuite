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

	bool print_log = false;
	int n_threads = -1;

	void printlog(String out_log);

protected:
	static void _bind_methods();

public:
	StableDiffusion();
	~StableDiffusion();

	void set_print_log(bool p_print_log);
	bool is_print_log() const;

	void set_n_threads(bool p_threads);
	int get_n_threads() const;

	Array get_vk_devices() const;
	int get_sys_physical_cores() const;
};

#endif // STABLE_DIFFUSION_H
