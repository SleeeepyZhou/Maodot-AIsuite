#ifndef AI_OBJECT_H
#define AI_OBJECT_H

#include "scene/main/node.h"

class AIObject : public Node {
	GDCLASS(AIObject, Node);

private:
	int count;

protected:
	static void _bind_methods();

public:
	AIObject();
	~AIObject();
	void add(int p_value);
	void reset();
	int get_total() const;
};

#endif // AI_OBJECT_H
