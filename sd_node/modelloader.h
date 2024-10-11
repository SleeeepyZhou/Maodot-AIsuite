#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"
#include "scene/main/node.h"

#include "stable-diffusion.h"

class ModelLoad : public StableDiffusion {
	GDCLASS(ModelLoad, StableDiffusion);

public:
	typedef schedule_t Scheduler;

private:
    sd_ctx_t* SDModel;
	Scheduler schedule = DEFAULT;
	String lora_path;

protected:
	static void _bind_methods();

public:
	ModelLoad();
	~ModelLoad();
	void load_model(String model_path);
    void free_model();
	void set_schedule(Scheduler p_schedule);
	Scheduler get_schedule() const;
};

#endif // MODEL_LOADER_H