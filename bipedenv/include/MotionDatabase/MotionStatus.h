#ifndef MOTIONSTATUS
#define MOTIONSTATUS
#include "MotionDatabase/MotionDB.h"
#include "Parameter.h"

template<typename T> class MotionStatus;
template<typename T> class MotionStatusLeaf;
template<typename T> using MotionStatusPtr = std::shared_ptr<MotionStatus<T>>;
template<typename T> using MotionStatusLeafPtr = std::shared_ptr<MotionStatusLeaf<T>>;

template<typename T>
class MotionStatus{
	public:
		MotionStatus(MotionDBPtr motionDB);
		virtual Eigen::VectorXd getPosition(T offset = 0) = 0;
		virtual Eigen::VectorXd getVelocity(T offset = 0);
		virtual bool isFootContact(int foot, T offset = 0);
		virtual int footContactInfo(T offset = 0) = 0;
		virtual MotionStatusPtr<T> step(MotionStatusPtr<T> self, T time = 1) = 0;
		virtual void translate(double dx, double dz) = 0;
		virtual bool operator != (const MotionStatusLeaf<T> &rhs) = 0;
		virtual bool isEnd(T offset = 0) = 0;
		virtual T remainFrame(T offset = 0) = 0;
		virtual double getPhase(T offset = 0) = 0;
		virtual double getTimestep() = 0;
		virtual MotionStatusPtr<T> clone() = 0;
		MotionDBPtr motionDB;
};

template<typename T>
class MotionStatusLeaf : public MotionStatus<T>{
	public:
		MotionStatusLeaf(MotionDBPtr motionDB, int clip, T frame, double Ydir, double Xpos, double Zpos);
		int clip;
		T frame;
		double Ydir, Xpos, Zpos; // offset
		
		virtual Eigen::VectorXd getPosition(T offset = 0);
		virtual Eigen::VectorXd getVelocity(T offset = 0);
		virtual int footContactInfo(T offset = 0);
		virtual MotionStatusPtr<T> step(MotionStatusPtr<T> self, T time = 1);
		virtual void translate(double dx, double dz);
		virtual bool operator != (const MotionStatusLeaf<T> &rhs);
		virtual bool isEnd(T offset);
		virtual T remainFrame(T offset = 0);
		virtual double getPhase(T offset);
		virtual double getTimestep();
		virtual MotionStatusPtr<T> clone();
		using MotionStatus<T>::motionDB;
};

template<typename T>
class MotionStatusBlend : public MotionStatus<T>{
	public:
		MotionStatusBlend(MotionDBPtr motionDB, MotionStatusPtr<T> from, MotionStatusPtr<T> to, T blendingFrame = Parameter::blendingFrame);
//		std::function<double(double)> speedControl;
		MotionStatusPtr<T> from, to;
		T blendingFrame, currentFrame;
		
		virtual Eigen::VectorXd getPosition(T offset = 0);
		virtual int footContactInfo(T offset = 0);
		virtual MotionStatusPtr<T> step(MotionStatusPtr<T> self, T time = 1);
		virtual void translate(double dx, double dz);
		virtual bool operator != (const MotionStatusLeaf<T> &rhs);
		virtual bool isEnd(T offset);
		virtual T remainFrame(T offset = 0);
		virtual double getPhase(T offset);
		virtual double getTimestep();
		virtual MotionStatusPtr<T> clone();
		using MotionStatus<T>::motionDB;
};
template<typename T> using MotionStatusBlendPtr = std::shared_ptr<MotionStatusBlend<T>>;

template<typename T>
class MotionStatusOffset : public MotionStatus<T>{
	public:
		MotionStatusOffset(MotionDBPtr motionDB, MotionStatusPtr<T> to, const Eigen::VectorXd &diff, T blendingFrame = Parameter::blendingFrame);
		MotionStatusPtr<T> to;
		Eigen::VectorXd diff;
		T blendingFrame, currentFrame;
		
		virtual Eigen::VectorXd getPosition(T offset = 0);
		virtual int footContactInfo(T offset = 0);
		virtual MotionStatusPtr<T> step(MotionStatusPtr<T> self, T time = 1);
		virtual void translate(double dx, double dz);
		virtual bool operator != (const MotionStatusLeaf<T> &rhs);
		virtual bool isEnd(T offset);
		virtual T remainFrame(T offset = 0);
		virtual double getPhase(T offset);
		virtual double getTimestep();
		virtual MotionStatusPtr<T> clone();
		using MotionStatus<T>::motionDB;
};
template<typename T> using MotionStatusOffsetPtr = std::shared_ptr<MotionStatusOffset<T>>;

typedef std::vector<std::pair<int, Eigen::Vector3d>> FootConstraints;
template<typename T>
class MotionStatusFootIK : public MotionStatus<T>{
	public:
		MotionStatusFootIK(MotionDBPtr motionDB, SkeletonPtr skel, MotionStatusPtr<T> from, FootConstraints footContact);
		SkeletonPtr skel;
		MotionStatusPtr<T> from;
		FootConstraints footContact;
		
		virtual Eigen::VectorXd getPosition(T offset = 0);
		virtual int footContactInfo(T offset = 0);
		virtual MotionStatusPtr<T> step(MotionStatusPtr<T> self, T time = 1);
		virtual void translate(double dx, double dz);
		virtual bool operator != (const MotionStatusLeaf<T> &rhs);
		virtual bool isEnd(T offset);
		virtual T remainFrame(T offset = 0);
		virtual double getPhase(T offset);
		virtual double getTimestep();
		virtual MotionStatusPtr<T> clone();
		using MotionStatus<T>::motionDB;
};
template<typename T> using MotionStatusFootIKPtr = std::shared_ptr<MotionStatusFootIK<T>>;

#endif
