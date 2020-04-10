#include "MotionDatabase/MotionStatus.h"
#include "MotionDatabase/MotionHelper.h"
#include "Helper/Functions.h"
#include "Parameter.h"

template<typename T>
MotionStatus<T>::MotionStatus(MotionDBPtr motionDB):motionDB(motionDB){
}

template<typename T>
Eigen::VectorXd MotionStatus<T>::getVelocity(T offset){
	__builtin_unreachable;
}

template<typename T>
int MotionStatus<T>::footContactInfo(T offset){
	__builtin_unreachable;
}

template<typename T>
bool MotionStatus<T>::isFootContact(int foot, T offset){
	return footContactInfo(offset) & 1<<foot;
}

// Leaf
template<typename T>
MotionStatusLeaf<T>::MotionStatusLeaf(MotionDBPtr motionDB, int clip, T frame, double Ydir, double Xpos, double Zpos):
	MotionStatus<T>(motionDB), clip(clip), frame(frame), Ydir(Ydir), Xpos(Xpos), Zpos(Zpos){
}

template<typename T>
Eigen::VectorXd MotionStatusLeaf<T>::getPosition(T offset){
	Eigen::VectorXd position = motionDB->getPosition(clip, frame + offset);
	positionRotateY(position, Ydir);
	position.segment(3, 3) += Eigen::Vector3d(Xpos, 0, Zpos);
	return position;
}

// This could be wrong since rotate Ydir - global velocity
template<typename T>
Eigen::VectorXd MotionStatusLeaf<T>::getVelocity(T offset){
	Eigen::VectorXd velocity = motionDB->getVelocity(clip, frame + offset);
	positionRotateY(velocity, Ydir);
	return velocity;
}

template<typename T>
int MotionStatusLeaf<T>::footContactInfo(T offset){
	return motionDB->footContactInfo(clip, frame + offset);
}

template<typename T>
MotionStatusPtr<T> MotionStatusLeaf<T>::step(MotionStatusPtr<T> self, T time){
	frame += time;
	return self;
}

template<typename T>
void MotionStatusLeaf<T>::translate(double dx, double dz){
	Xpos += dx; Zpos += dz;
}

template<typename T>
bool MotionStatusLeaf<T>::operator != (const MotionStatusLeaf<T> &rhs){
	return clip != rhs.clip || abs(frame - rhs.frame) > 30;
}

template<typename T>
bool MotionStatusLeaf<T>::isEnd(T offset){
	return motionDB->numFrame(clip) <= frame + offset + 1;
}

template<typename T>
T MotionStatusLeaf<T>::remainFrame(T offset){
	return motionDB->numFrame(clip) - (frame + offset + 1);
}

template<typename T>
double MotionStatusLeaf<T>::getPhase(T offset){
	return fmod((double)(frame + offset) / (motionDB->numFrame(clip)-1), 1);
}

template<typename T>
double MotionStatusLeaf<T>::getTimestep(){
	return 1. / motionDB->numFrame(clip);
}

template<typename T>
MotionStatusPtr<T> MotionStatusLeaf<T>::clone(){
	return MotionStatusPtr<T>(new MotionStatusLeaf<T>(motionDB, clip, frame, Ydir, Xpos, Zpos));
}

// Blend
template<typename T>
MotionStatusBlend<T>::MotionStatusBlend(MotionDBPtr motionDB, MotionStatusPtr<T> from, MotionStatusPtr<T> to, T blendingFrame):
	MotionStatus<T>(motionDB), from(from), to(to), blendingFrame(blendingFrame), currentFrame(0){
}

template<typename T>
Eigen::VectorXd MotionStatusBlend<T>::getPosition(T offset){
	Eigen::VectorXd fromPosition = from->getPosition(offset),
		toPosition = to->getPosition(offset);
	T frame = currentFrame + offset;
	if(frame <= 0) return fromPosition;
	else if(frame >= blendingFrame) return toPosition;
	else return blendingPosition(fromPosition, toPosition, frame, blendingFrame);
}

template<typename T>
int MotionStatusBlend<T>::footContactInfo(T offset){
	return from->footContactInfo(offset) & to->footContactInfo(offset);
}

template<typename T>
MotionStatusPtr<T> MotionStatusBlend<T>::step(MotionStatusPtr<T> self, T time){
	currentFrame += time;
	to = to->step(to, time);
	if(currentFrame >= Parameter::blendingFrame) return to;
	else{ from = from->step(from, time); return self; }
}

template<typename T>
void MotionStatusBlend<T>::translate(double dx, double dz){
	from->translate(dx, dz);
	to->translate(dx, dz);
}

template<typename T>
bool MotionStatusBlend<T>::operator != (const MotionStatusLeaf<T> &rhs){
	return *to != rhs;
}

template<typename T>
bool MotionStatusBlend<T>::isEnd(T offset){
	return from->isEnd(offset) && to->isEnd(offset);
}

template<typename T>
T MotionStatusBlend<T>::remainFrame(T offset){
	return to->remainFrame();
}

template<typename T>
double MotionStatusBlend<T>::getPhase(T offset){
	return to->getPhase(offset);
}

template<typename T>
double MotionStatusBlend<T>::getTimestep(){
	assert(to->getTimestep() == from->getTimestep());
	return to->getTimestep();
}

template<typename T>
MotionStatusPtr<T> MotionStatusBlend<T>::clone(){
	return MotionStatusPtr<T>(new MotionStatusBlend<T>(motionDB, from->clone(), to->clone()));
}

// Offset
template<typename T>
MotionStatusOffset<T>::MotionStatusOffset(MotionDBPtr motionDB, MotionStatusPtr<T> to, const Eigen::VectorXd &diff, T blendingFrame):
	MotionStatus<T>(motionDB), to(to), diff(diff), blendingFrame(blendingFrame), currentFrame(0){
}

template<typename T>
Eigen::VectorXd MotionStatusOffset<T>::getPosition(T offset){
	Eigen::VectorXd toPosition = to->getPosition(offset);
	T frame = currentFrame + offset;
	if(frame < 0) frame = 0;
	else if(frame >= blendingFrame) return toPosition;

	int sz = toPosition.size();
	double rate = 1 - frame / (double)blendingFrame;
	rate = (sin(rate*PI - PI/2) + 1) / 2;
	Eigen::VectorXd result = Eigen::VectorXd::Zero(sz);
	// position
	for(int t = 3; t <= 3; t += 3){
		result.segment(t, 3) = toPosition.segment(t, 3) + diff.segment(t, 3) * rate;
	}
	// rotation
	for(int i = 3; i < sz; i += 3){
		int t = i == 3? 0 : i;
		Eigen::Quaterniond to = DARTPositionToQuaternion(toPosition.segment(t, 3));
		Eigen::AngleAxisd tmp = DARTPositionToAngleAxisd(diff.segment(t, 3));
		tmp.angle() *= rate;
		result.segment(t, 3) = QuaternionToDARTPosition(to * tmp);
	}
	return result;
}

template<typename T>
int MotionStatusOffset<T>::footContactInfo(T offset){
	return to->footContactInfo(offset);
}

template<typename T>
MotionStatusPtr<T> MotionStatusOffset<T>::step(MotionStatusPtr<T> self, T time){
	currentFrame += time;
	to = to->step(to, time);
	if(currentFrame >= blendingFrame) return to;
	else return self;
}

template<typename T>
void MotionStatusOffset<T>::translate(double dx, double dz){
	to->translate(dx, dz);
}

template<typename T>
bool MotionStatusOffset<T>::operator != (const MotionStatusLeaf<T> &rhs){
	return *to != rhs;
}

template<typename T>
bool MotionStatusOffset<T>::isEnd(T offset){
	return to->isEnd(offset);
}

template<typename T>
T MotionStatusOffset<T>::remainFrame(T offset){
	return to->remainFrame();
}

template<typename T>
double MotionStatusOffset<T>::getPhase(T offset){
	return to->getPhase(offset);
}

template<typename T>
double MotionStatusOffset<T>::getTimestep(){
	return to->getTimestep();
}

template<typename T>
MotionStatusPtr<T> MotionStatusOffset<T>::clone(){
	return MotionStatusPtr<T>(new MotionStatusOffset<T>(motionDB, to->clone(), diff));
}

// FootIK
template<typename T>
MotionStatusFootIK<T>::MotionStatusFootIK(MotionDBPtr motionDB, SkeletonPtr skel, MotionStatusPtr<T> from, FootConstraints footContact):
	MotionStatus<T>(motionDB), skel(skel), from(from), footContact(footContact){
}

template<typename T>
Eigen::VectorXd MotionStatusFootIK<T>::getPosition(T offset){
	Eigen::VectorXd fromPosition = from->getPosition(offset);
	skel->setPositions(fromPosition);
	skel->computeForwardKinematics(true, false, false);
	for(auto [idx, constraint] : footContact){
		BodyNodePtr node = skel->getBodyNode(Parameter::footList[idx]);
		// TODO : IK, multiple foot contact
		auto IKmodule = node->getIK(true);
		Eigen::Isometry3d target = node->getWorldTransform(); target.translation() = constraint;
		IKmodule->setTarget(SimpleFramePtr(new SimpleFrame(Frame::World(), Parameter::footList[idx], target)));
		IKmodule->solveAndApply();
	}
	return skel->getPositions();
}

template<typename T>
int MotionStatusFootIK<T>::footContactInfo(T offset){
	int result = from->footContactInfo(offset);
	for(auto constraint : footContact) result |= 1 << constraint.first;
	return result;
}

template<typename T>
MotionStatusPtr<T> MotionStatusFootIK<T>::step(MotionStatusPtr<T> self, T time){
	Eigen::VectorXd currentPosition = getPosition(), nextPosition;
	FootConstraints remain;
	MotionStatusPtr<T> nxt;
	for(auto constraint : footContact){
		if(from->isFootContact(constraint.first, time)) remain.push_back(constraint);
	}
	if(footContact.size() != remain.size()){
		footContact = remain;
		nextPosition = getPosition();
		Eigen::VectorXd diff = skel->getPositionDifferences(currentPosition, nextPosition);
		from = MotionStatusOffsetPtr<T>(new MotionStatusOffset<T>(motionDB, from, diff));
		from = from->step(from, time);
		if(remain.size()) return self;
		else return from;
	}
	else{
		from = from->step(from, time);
		return self;
	}
}

template<typename T>
void MotionStatusFootIK<T>::translate(double dx, double dz){
	from->translate(dx, dz);
	for(auto &constraint : footContact) constraint.second += Eigen::Vector3d(dx, 0, dz);
}

template<typename T>
bool MotionStatusFootIK<T>::operator != (const MotionStatusLeaf<T> &rhs){
	return *from != rhs;
}

template<typename T>
bool MotionStatusFootIK<T>::isEnd(T offset){
	return from->isEnd(offset);
}

template<typename T>
T MotionStatusFootIK<T>::remainFrame(T offset){
	return from->remainFrame();
}

template<typename T>
double MotionStatusFootIK<T>::getPhase(T offset){
	return from->getPhase(offset);
}

template<typename T>
double MotionStatusFootIK<T>::getTimestep(){
	return from->getTimestep();
}

template<typename T>
MotionStatusPtr<T> MotionStatusFootIK<T>::clone(){
	return MotionStatusPtr<T>(new MotionStatusFootIK<T>(motionDB, skel, from->clone(), footContact));
}

template class MotionStatus<int>;
template class MotionStatusLeaf<int>;
template class MotionStatusOffset<int>;
template class MotionStatusBlend<int>;
template class MotionStatusFootIK<int>;

template class MotionStatus<double>;
template class MotionStatusLeaf<double>;
template class MotionStatusOffset<double>;
template class MotionStatusBlend<double>;
template class MotionStatusFootIK<double>;