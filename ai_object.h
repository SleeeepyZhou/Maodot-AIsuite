#ifndef AI_OBJECT_H
#define AI_OBJECT_H

#include "scene/main/node.h"
#include "core/io/resource.h"

class AIResource : public Resource {
	GDCLASS(AIResource, Resource);

protected:
	static void _bind_methods();

public:
	AIResource();
	~AIResource();
};

class AIObject : public Node {
	GDCLASS(AIObject, Node);

	bool print_log = false;
	void printlog(String out_log);

protected:
	static void _bind_methods();

public:
	AIObject();
	~AIObject();

	void set_print_log(bool p_print_log);
	bool is_print_log() const;
	int get_sys_physical_cores() const;
};

#endif // AI_OBJECT_H
