/*       By SleeeepyZhou        */

#ifndef COND_H
#define COND_H

#include "stablediffusion.h"

class SDCond : public SDResource {
	GDCLASS(SDCond, SDResource);

private:
    int clip_skip = -1;

protected:
	static void _bind_methods();

public:
    SDCond();
    ~SDCond();
};

class Control : public StableDiffusion {
	GDCLASS(Control, StableDiffusion);

private:
    CLIP clip_res;

protected:
    static void _bind_methods();

public:
    Control();
    ~Control();
    SDCond text_encoders(CLIP clip_res, String prompt);
}

#endif // COND_H