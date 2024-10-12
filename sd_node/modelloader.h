#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "stablediffusion.h"

#include "stable-diffusion.h"

class SDModel : public SDResource {
	GDCLASS(SDModel, SDResource);

	String model_path

	

protected:
	static void _bind_methods();

public:
    SDModel();
    ~SDModel();
    void set_model(const String &p_model_path);
	void _set_model_path(const String &p_model_path);
	String _get_model_path() const;
};


class SDModelLoader : public StableDiffusion {
	GDCLASS(SDModelLoader, StableDiffusion);

public:
	typedef schedule_t Scheduler;

private:
    sd_ctx_t* SDModel;
	Scheduler schedule = DEFAULT;
	String lora_path;

protected:
	static void _bind_methods();

public:
	SDModelLoader();
	~SDModelLoader();
	void load_model(String model_path);
    void free_model();
	void set_schedule(Scheduler p_schedule);
	Scheduler get_schedule() const;
};

#endif // MODEL_LOADER_H