#include <Eigen/Dense>
#include<dart/dart.hpp>
#include "Parameter.h"
#include "Environment/EnvironmentHelper.h"
#include "Helper/Functions.h"

double getDeepMimicReward(SkeletonPtr physicsSkel, SkeletonPtr kinematicsSkel){
	// Position and velocities differences
	Eigen::VectorXd p_diff = physicsSkel->getPositionDifferences(kinematicsSkel->getPositions(), physicsSkel->getPositions());
	Eigen::VectorXd v_diff = physicsSkel->getPositionDifferences(kinematicsSkel->getVelocities(), physicsSkel->getVelocities());
	Eigen::VectorXd p_diff_reward(Parameter::rewardBodies.size() * 3);
	Eigen::VectorXd v_diff_reward(Parameter::rewardBodies.size() * 3);

	for(int i = 0; i < (int)Parameter::rewardBodies.size(); i++){
		int idx = physicsSkel->getBodyNode(Parameter::rewardBodies[i])->getParentJoint()->getIndexInSkeleton(0);
		p_diff_reward.segment<3>(3*i) = p_diff.segment<3>(idx);
		v_diff_reward.segment<3>(3*i) = v_diff.segment<3>(idx);
	}

	// COM differences
	Eigen::Vector3d com_diff = physicsSkel->getCOM() - kinematicsSkel->getCOM();

	// End-effector position differences
	Eigen::VectorXd ee_diff(Parameter::endEffectors.size() * 3);
	for (int i = 0; i < (int)Parameter::endEffectors.size(); i++){
		Eigen::Isometry3d diff = physicsSkel->getBodyNode(Parameter::endEffectors[i])->getWorldTransform().inverse() * 
			kinematicsSkel->getBodyNode(Parameter::endEffectors[i])->getWorldTransform();
		ee_diff.segment<3>(3 * i) = diff.translation();
	}

	// Evaluate total reward
	double scale = 1.0;
	double sig_p = 0.1 * scale;   // 2
	double sig_v = 1.0 * scale;   // 3
	double sig_com = 0.3 * scale; // 4
	double sig_ee = 0.3 * scale;  // 8

	double r_p = DPhy::exp_of_squared(p_diff_reward, sig_p);
	double r_v = DPhy::exp_of_squared(v_diff_reward, sig_v);
	double r_com = DPhy::exp_of_squared(com_diff, sig_com);
	double r_ee = DPhy::exp_of_squared(ee_diff, sig_ee);

	double r_tot = r_p*r_v*r_com*r_ee;
	if(dart::math::isNan(r_tot)) return 0;
	return r_tot;
}

bool getDeepMimicEarlyTerminate(SkeletonPtr physicsSkel, SkeletonPtr kinematicsSkel, int flag){
	// Nan check
	Eigen::VectorXd position = physicsSkel->getPositions();
	Eigen::VectorXd velocity = physicsSkel->getVelocities();
	if(dart::math::isNan(position) || dart::math::isNan(velocity)){
		return true;
	}

	// Early termination
	// Height limit
	double root_y = position[4];
	if (flag&ROOT_HEIGHT && (root_y < Parameter::rootHeightLowerLimit || root_y > Parameter::rootHeightUpperLimit)){
//		this->mTerminationReason = TerminationReason::ROOT_HEIGHT;
		return true;
	}

	// root distance limit
	Eigen::Isometry3d root_diff = physicsSkel->getRootBodyNode()->getWorldTransform().inverse()
		* kinematicsSkel->getRootBodyNode()->getWorldTransform();
	Eigen::Vector3d root_pos_diff = root_diff.translation();
	if (flag&ROOT_DIFF && (root_pos_diff.norm() > Parameter::rootDiffThreshold)){
//		this->mTerminationReason = TerminationReason::ROOT_DIFF;
		return true;
	}

	// root rotation limit
	Eigen::AngleAxisd root_diff_aa(root_diff.linear());
	double angle = DPhy::RadianClamp(Eigen::AngleAxisd(root_diff.linear()).angle());
	if (flag&ROOT_ANGLE_DIFF && (std::abs(angle) > Parameter::rootAngleDiffThreshold)){
//		this->mTerminationReason = TerminationReason::ROOT_ANGLE_DIFF;
		return true;
	}
	return false;
}
