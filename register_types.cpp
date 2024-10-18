/* register_types.cpp */

#include "core/object/class_db.h"
#include "register_types.h"

/* Module headers */
#include "ai_object.h"
#include "stablediffusion.h"

#include "sdmodel.h"
#include "sdcond.h"
#include "ksampler.h"
#include "latent.h"

void initialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_ABSTRACT_CLASS(AIObject);
	GDREGISTER_ABSTRACT_CLASS(StableDiffusion);
	
	GDREGISTER_CLASS(SDModel);

	GDREGISTER_CLASS(SDCond);
	GDREGISTER_CLASS(SDControl);
	
	GDREGISTER_CLASS(Latent);

	GDREGISTER_ABSTRACT_CLASS(AIResource);
	GDREGISTER_ABSTRACT_CLASS(SDResource);
	GDREGISTER_ABSTRACT_CLASS(StableDiffusionGGML);

}

void uninitialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
