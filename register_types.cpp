/* register_types.cpp */

#include "core/object/class_db.h"

#include "ai_object.h"
#include "stablediffusion.h"

#include "register_types.h"

void initialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<AIObject>();
	ClassDB::register_class<StableDiffusion>();
}

void uninitialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
