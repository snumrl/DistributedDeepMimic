#include "MotionDatabase/MotionHelper.h"
#include "Helper/Functions.h"
#include "Parameter.h"

bool isFootContact(dart::dynamics::BodyNodePtr n){
	return n->getWorldTransform().translation()[1] < Parameter::footContactHeight && n->getLinearVelocity().norm() < Parameter::footContactVelocity;
}

Eigen::VectorXd blendingPosition(Eigen::VectorXd fromPosition, Eigen::VectorXd toPosition, double frame, double blendingFrame){
	int sz = fromPosition.size();
	double rate = frame / blendingFrame;
	Eigen::VectorXd result = Eigen::VectorXd::Zero(sz);
	// position
	for(int t = 3; t <= 3; t += 3){
		Eigen::Vector3d from = fromPosition.segment(t, 3),
			to = toPosition.segment(t, 3);
		result.segment(t, 3) = from * (1 - rate) + to * rate;
	}
	// rotation
	for(int i = 3; i < sz; i += 3){
		int t = i == 3? 0 : i;
		Eigen::Quaterniond from = DARTPositionToQuaternion(fromPosition.segment(t, 3)),
			to = DARTPositionToQuaternion(toPosition.segment(t, 3));
		result.segment(t, 3) = QuaternionToDARTPosition(from.slerp(rate, to));
	}
	return result;
}

template<typename T>
void setMotionAlignment(MotionStatusLeafPtr<T> current, Eigen::Vector6d target){
	Eigen::VectorXd pos = current->motionDB->getPosition(current->clip, current->frame).segment(0, 6);
	current->Ydir = getYrot(pos.segment(0, 3), target.segment(0, 3));
	positionRotateY(pos, current->Ydir);
	current->Xpos = target[3] - pos[3];
	current->Zpos = target[5] - pos[5];
}

template<typename T>
MotionStatusPtr<T> getMotionStatusOffset(MotionStatusPtr<T> to, Eigen::VectorXd currentPosition, SkeletonPtr skel){
	Eigen::VectorXd diff = skel->getPositionDifferences(currentPosition, to->getPosition());
	return MotionStatusPtr<T>(new MotionStatusOffset<T>(to->motionDB, to, diff));
}

template<typename T>
MotionStatusPtr<T> getMotionStatusFootIK(MotionStatusPtr<T> from, SkeletonPtr skel, SkeletonPtr skelcopy){
	FootConstraints footContact;
	for(int i = 0; i < (int) Parameter::footList.size(); i++){
		BodyNodePtr node = skel->getBodyNode(Parameter::footList[i]);
		if(isFootContact(node)) footContact.emplace_back(i, node->getWorldTransform().translation());
	}
	if(footContact.size() == 0) return from;
	return MotionStatusFootIKPtr<T>(new MotionStatusFootIK<T>(from->motionDB, skelcopy, from, footContact));
}

template void setMotionAlignment(MotionStatusLeafPtr<int>, Eigen::Vector6d);
template MotionStatusPtr<int> getMotionStatusOffset(MotionStatusPtr<int>, Eigen::VectorXd, SkeletonPtr);
template MotionStatusPtr<int> getMotionStatusFootIK(MotionStatusPtr<int>, SkeletonPtr, SkeletonPtr);

template void setMotionAlignment(MotionStatusLeafPtr<double>, Eigen::Vector6d);
template MotionStatusPtr<double> getMotionStatusOffset(MotionStatusPtr<double>, Eigen::VectorXd, SkeletonPtr);
template MotionStatusPtr<double> getMotionStatusFootIK(MotionStatusPtr<double>, SkeletonPtr, SkeletonPtr);