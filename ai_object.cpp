#include "ai_object.h"

AIResource::AIResource() {
}

AIResource::~AIResource() {
}

void AIResource::_bind_methods() {
}

AIObject::AIObject() {
}
AIObject::~AIObject() {
}

void AIObject::printlog(String out_log) {
    if (print_log) {
        print_line(out_log);
    }
    emit_signal(SNAME("sd_log"), out_log);
}

void AIObject::set_print_log(bool p_print_log) {
    print_log = p_print_log;
}
bool AIObject::is_print_log() const {
	return print_log;
}

int AIObject::get_sys_physical_cores() const {
    return DeviceInfo::getInstance().get_core_count();
}

void AIObject::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_print_log", "enable"), &AIObject::set_print_log);
    ClassDB::bind_method(D_METHOD("is_print_log"), &AIObject::is_print_log);
    ClassDB::bind_method(D_METHOD("get_sys_physical_cores"), &AIObject::get_sys_physical_cores);

    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "print_log"), "set_print_log", "is_print_log");

    ADD_SIGNAL(MethodInfo("sd_log", PropertyInfo(Variant::STRING, "SD_log")));
}
