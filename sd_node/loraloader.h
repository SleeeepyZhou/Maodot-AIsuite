#ifndef LORA_LOADER_H
#define LORA_LOADER_H

#include "stablediffusion.h"

class LoraLoader : public StableDiffusion {
	GDCLASS(LoraLoader, StableDiffusion);

private:
	String lora_model_dir;


}

#endif // LORA_LOADER_H