/* register_types.cpp */

#include "core/object/class_db.h"
#include "register_types.h"

/* Module headers */
#include "ai_object.h"
#include "stablediffusion.h"

#include "modelloader.h"
#include "sdcond.h"
#include "ksampler.h"
#include "latent.h"
#include "vae_node.h"

void initialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_ABSTRACT_CLASS(AIObject);
	GDREGISTER_ABSTRACT_CLASS(StableDiffusion);
	
	GDREGISTER_CLASS(SDModelLoader);
	GDREGISTER_CLASS(SDControl);
	GDREGISTER_CLASS(KSampler);
	GDREGISTER_CLASS(VAE);

	GDREGISTER_ABSTRACT_CLASS(AIResource);
	GDREGISTER_ABSTRACT_CLASS(SDResource);
	GDREGISTER_ABSTRACT_CLASS(SDModel);

	GDREGISTER_CLASS(Backend);
	GDREGISTER_CLASS(CLIP);
	GDREGISTER_CLASS(Diffusion);
	GDREGISTER_CLASS(VAEModel);

	GDREGISTER_CLASS(SDCond);
	GDREGISTER_CLASS(Latent);
}

void uninitialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
