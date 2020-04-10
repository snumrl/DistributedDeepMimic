#include "Render/Shader.h"
#include "Render/GLFWMain.h"
#include "Agents/PhysicsSPDAgent.h"
#include "Agents/DefaultAgent.h"
#include "Render/DART_interface.h"
#include "MotionDatabase/SkeletonBuilder.h"
#include "Helper/Functions.h"
#include "Parameter.h"
#include "Helper/TimeProfiler.h"
#include<vector>
#include<tuple>

PhysicsSPDAgent::PhysicsSPDAgent() : PhysicsAgent(){
	Eigen::VectorXd tmp = Eigen::VectorXd::Ones(skel->getNumDofs()); tmp.head<6>().setZero();
	mKp = tmp * Parameter::mKp;
	mKv = tmp * Parameter::mKv;
}

Eigen::VectorXd PhysicsSPDAgent::getSPDForces(const Eigen::VectorXd &targetPosition, const Eigen::VectorXd &targetVelocity)
{
	Eigen::VectorXd q = skel->getPositions();
	Eigen::VectorXd dq = skel->getVelocities();
	double dt = skel->getTimeStep();
	Eigen::MatrixXf M = (skel->getMassMatrix() + Eigen::MatrixXd(dt * mKv.asDiagonal())).cast<float>();
	Eigen::MatrixXd M_inv = M.inverse().cast<double>();

	Eigen::VectorXd p_d = q + dq*dt - targetPosition;
	// clamping radians to [-pi, pi], only for ball joints
	// TODO : make it for all type joints
	/*
	p_d.segment<6>(0) = Eigen::VectorXd::Zero(6);
	for (int i = 6; i < (int)skel->getNumDofs(); i += 3)
	{
		Eigen::Quaterniond q_s = DPhy::DARTPositionToQuaternion(q.segment<3>(i));
		Eigen::Quaterniond dq_s = DPhy::DARTPositionToQuaternion(dt * (dq.segment<3>(i)));
		Eigen::Quaterniond q_d_s = DPhy::DARTPositionToQuaternion(targetPosition.segment<3>(i));

		Eigen::Quaterniond p_d_s = q_d_s.inverse() * q_s * dq_s;

		Eigen::Vector3d v = DPhy::QuaternionToDARTPosition(p_d_s);
		double angle = v.norm();
		if (angle > 1e-8)
		{
			Eigen::Vector3d axis = v.normalized();

			angle = DPhy::RadianClamp(angle);
			p_d.segment<3>(i) = angle * axis;
		}
		else
			p_d.segment<3>(i) = v;
	} // */
	Eigen::VectorXd p_diff = -mKp.cwiseProduct(p_d);
	Eigen::VectorXd v_diff = -mKv.cwiseProduct(dq - targetVelocity);
	Eigen::VectorXd qddot = M_inv * (-skel->getCoriolisAndGravityForces() +
		p_diff + v_diff + skel->getConstraintForces());

	Eigen::VectorXd tau = p_diff + v_diff - dt * mKv.cwiseProduct(qddot);
	tau.segment<6>(0) = Eigen::VectorXd::Zero(6);

	return tau;
}

void PhysicsSPDAgent::step(const Eigen::VectorXd &torque){
	PhysicsAgent::step(torque);
}

void PhysicsSPDAgent::step(const Eigen::VectorXd &targetPosition, const Eigen::VectorXd &targetVelocity, int repeat){
	Eigen::VectorXd torque = getSPDForces(targetPosition, targetVelocity);
	for(int i = 0; i < repeat; i++) PhysicsAgent::step(torque);
}
