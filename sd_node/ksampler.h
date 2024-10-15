/*       By SleeeepyZhou        */

#ifndef K_SAMPLER_H
#define K_SAMPLER_H

#include "stablediffusion.h"

#include "ggml_extend.hpp"

/* Latent */
class Latent : public SDResource {
	GDCLASS(Latent, SDResource);

private:
    int width = 512;
	int height = 512;
    int batch_count = 1;

    ggml_tensor* latent = NULL;
    struct ggml_context* work_ctx = NULL;

protected:
	static void _bind_methods();

public:
    Latent();
    ~Latent();
	void set_width(const int &p_width);
	int get_width() const;
    void set_height(const int &p_height);
	int get_height() const;

    void create_latent(SDVersion version);
    ggml_tensor* get_latent() const;
    struct ggml_context* get_work_ctx() const;

};

/* KSampler */
class KSampler : public StableDiffusion {
	GDCLASS(KSampler, StableDiffusion);
    
    NodePath modelloader;
    
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
    void sample(Latent init_latent);
    
};

#endif // K_SAMPLER_H
