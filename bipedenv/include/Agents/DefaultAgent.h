#ifndef DEFAULTAGENT
#define DEFAULTAGENT

#include "Render/Drawable.h"
#include <dart/dart.hpp>

using namespace dart::dynamics;
using namespace dart::simulation;

class KinematicsController;
class DefaultAgent : public Drawable{
	public:
		DefaultAgent(SkeletonPtr skel, const Eigen::Vector3d &color);
		SkeletonPtr skel;

		// For rendering
		virtual void display();
        Eigen::Vector3d color;
};

using DefaultAgentPtr = std::shared_ptr<DefaultAgent>;

#endif
