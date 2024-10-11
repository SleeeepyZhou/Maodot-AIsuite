#ifndef VAE_NODE_H
#define VAE_NODE_H

#include "stablediffusion.h"

#include "vae.hpp"
#include "tae.hpp"

class VAE : public StableDiffusion {
	GDCLASS(VAE, StableDiffusion);

public:
	typedef schedule_t Scheduler;

private:
    sd_ctx_t* SDModel;
	Scheduler schedule = DEFAULT;
	String lora_path;

protected:
	static void _bind_methods();

public:
	VAE();
	~VAE();
	void load_model(String model_path);
    void free_model();
	void set_schedule(Scheduler p_schedule);
	Scheduler get_schedule() const;
}

#endif // VAE_NODE_H