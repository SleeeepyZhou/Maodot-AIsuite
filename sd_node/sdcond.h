/*       By SleeeepyZhou        */

#ifndef COND_H
#define COND_H

#include "stablediffusion.h"
#include "modelloader.h"

class SDCond : public SDResource {
	GDCLASS(SDCond, SDResource);

private:
    SDCondition cond;
    SDCondition uncond;

protected:
	static void _bind_methods();

public:
    SDCond(SDCondition cond, SDCondition uncond);
    ~SDCond();

    SDCondition get_cond() const;
    SDCondition get_uncond() const;
};

class SDControl : public StableDiffusion {
	GDCLASS(SDControl, StableDiffusion);

private:
    SDCond sdcond;

protected:
    static void _bind_methods();

public:
    SDControl();
    ~SDControl();
    SDCond get_cond_res() const;
	void text_encoders(SDModel model_node, Latent latent, 
                       String prompt, String negative_prompt, 
                       int clip_skip);
}

#endif // COND_H