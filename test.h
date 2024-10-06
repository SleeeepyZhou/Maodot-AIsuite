#ifndef AI_SUITE_H
#define AI_SUITE_H

#include "scene/main/node.h"

class AISuite : public Node {
	GDCLASS(AISuite, Node);

private:
	int count;

protected:
	static void _bind_methods();

public:
	TestNode();
	~TestNode();
	void add(int p_value);
	void reset();
	int get_total() const;
};

#endif // TEST_NODE_H
