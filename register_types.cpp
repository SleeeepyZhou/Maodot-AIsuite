/* register_types.cpp */

#include "register_types.h"

#include "core/object/class_db.h"
#include "ai_object.h"

void initialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	ClassDB::register_class<AIObject>();
}

void uninitialize_ai_suite_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
}
