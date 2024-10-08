#ifndef AI_OBJECT_H
#define AI_OBJECT_H

#include "scene/main/node.h"

class AIObject : public Node {
	GDCLASS(AIObject, Node);

protected:
	static void _bind_methods();

public:
	AIObject();
	~AIObject();
};

#endif // AI_OBJECT_H
