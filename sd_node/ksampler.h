/*       By SleeeepyZhou        */

#ifndef K_SAMPLER_H
#define K_SAMPLER_H

#include "stablediffusion.h"
#include "modelloader.h"
#include "latent.h"

#include "ggml_extend.hpp"
#include "rng.hpp"
#include "rng_philox.hpp"

/* KSampler */
class KSampler : public StableDiffusion {
	GDCLASS(KSampler, StableDiffusion);
    
public:
    enum SamplerName {
        EULER_A,
        EULER,
        HEUN,
        DPM2,
        DPMPP2S_A,
        DPMPP2M,
        DPMPP2Mv2,
        IPNDM,
        IPNDM_V,
        LCM,
        N_SAMPLE_METHODS
    };

private:
    int seed = 42;
    int steps = 10;
    float CFG = 8.0f;
    // float miniCFG;
    float denoise = 1.0f;
    SamplerName sampler_name = LCM;
    
protected:
	static void _bind_methods();

public:
	KSampler();
	~KSampler();
    void sample(Diffusion diffusion_model, Latent init_latent, SDCond cond, SDCond uncond);

};

#endif // K_SAMPLER_H
