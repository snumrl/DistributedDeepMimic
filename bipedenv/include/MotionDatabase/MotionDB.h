#ifndef MOTIONDB
#define MOTIONDB

#include "Render/Drawable.h"
#include<dart/dart.hpp>
#include<vector>
#include<string>

using namespace dart::dynamics;

class MotionDB : public Drawable{
	public:
		MotionDB(std::vector<std::string> files, SkeletonPtr skel);

		int numClip();
		int numFrame(int clip);

		int footContactInfo(int clip, int frame);
		int footContactInfo(int clip, double frame);
		bool isFootContact(int clip, int frame, int foot);
		bool isFootContact(int clip, double frame, int foot);
		Eigen::VectorXd getPosition(int clip, int frame);
		Eigen::VectorXd getPosition(int clip, double frame);
		Eigen::VectorXd getVelocity(int clip, int frame);
		Eigen::VectorXd getVelocity(int clip, double frame);

		std::vector<std::vector<Eigen::VectorXd>> motionData;
		std::vector<std::vector<int>> footContactData;

		std::vector<std::pair<int,int>> visitList;
		double width, height;
		void display();
		void reshape(int width, int height);
private:
		SkeletonPtr skel;
};

using MotionDBPtr = std::shared_ptr<MotionDB>;

#endif
