#ifndef PHYSICSAGENT
#define PHYSICSAGENT

#include "Agents/DefaultAgent.h"
#include <dart/dart.hpp>

using namespace dart::dynamics;
using namespace dart::simulation;

class PhysicsAgent : public DefaultAgent{
	public:
		PhysicsAgent();
		
		// For physics
        WorldPtr mWorld;
		
		// For action
		virtual void step(const Eigen::VectorXd &torque);
		virtual void reset(const Eigen::VectorXd &startPosition, const Eigen::VectorXd &startVelocity);
		Eigen::VectorXd getPosition();
		Eigen::VectorXd getVelocity();

		static WorldPtr physicsWorld;
		static WorldPtr getPhysicsWorld();

		// For rendering
		virtual void display();
};

using PhysicsAgentPtr = std::shared_ptr<PhysicsAgent>;

#endif
