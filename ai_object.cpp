#include "ai_object.h"

AIObject::AIObject() {
	count = 0;
}

AIObject::~AIObject() {
}

void AIObject::add(int p_value) {
	count += p_value;
}

void AIObject::reset() {
	count = 0;
}

int AIObject::get_total() const {
	return count;
}

void AIObject::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add", "value"), &AIObject::add);
	ClassDB::bind_method(D_METHOD("reset"), &AIObject::reset);
	ClassDB::bind_method(D_METHOD("get_total"), &AIObject::get_total);
}
