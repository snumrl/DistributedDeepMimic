#ifndef PHYSICSSPDAGENT
#define PHYSICSSPDAGENT

#include "Agents/PhysicsAgent.h"
#include <dart/dart.hpp>

using namespace dart::dynamics;
using namespace dart::simulation;

class PhysicsSPDAgent : public PhysicsAgent{
	public:
		PhysicsSPDAgent();
		
		// For physics
		Eigen::VectorXd mKp, mKv;

		// For action
		virtual void step(const Eigen::VectorXd &torque);
		virtual void step(const Eigen::VectorXd &targetPosition, const Eigen::VectorXd &targetVelocity, int repeat = 1);
		Eigen::VectorXd getSPDForces(const Eigen::VectorXd &targetPosition, const Eigen::VectorXd &targetVelocity);
};

using PhysicsSPDAgentPtr = std::shared_ptr<PhysicsSPDAgent>;

#endif
