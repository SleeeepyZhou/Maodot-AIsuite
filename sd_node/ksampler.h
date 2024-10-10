#ifndef K_SAMPLER_H
#define K_SAMPLER_H

#include "stablediffusion.h"
#include "scene/main/node.h"

#include "stable-diffusion.h"


class KSampler : public StableDiffusion {
	GDCLASS(KSampler, StableDiffusion);

public:
    typedef sample_method_t SamplerName

private:
    int seed = 42;
    int steps = 10;
    float CFG = 8.0;
    // float miniCFG;
    float denoise = 1.0;
    SamplerName sampler_name = LCM;
    

protected:
	static void _bind_methods();

public:
	KSampler();
	~KSampler();
    
};

#endif // K_SAMPLER_H

	// Ref<Image> t2i(String model_path, String prompt);
 /* #include "core/io/image.h"
	Ref<Image> t2i(String model_path, String prompt);
    */