/* register_types.cpp */

#include "core/object/class_db.h"

#include "register_types.h"

/* Module headers */
#include "ai_object.h"
#include "stablediffusion.h"
#include "modelloader.h"


void initialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_ABSTRACT_CLASS(AIObject);
	GDREGISTER_ABSTRACT_CLASS(StableDiffusion);
	GDREGISTER_CLASS(ModelLoad);
}

void uninitialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
