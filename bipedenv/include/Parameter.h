#ifndef MMPARAMETER
#define MMPARAMETER

#include<vector>
#include<string>

#define PARAM(MACRO) \
	/* Skeleton file */ \
	MACRO(std::string, humanoidFile) \
	/* Controller parameter */ \
	MACRO(double, smoothTime) \
	MACRO(double, TrajectoryWeight) \
	MACRO(double, minimunVelocity) \
	/* SPD */ \
	MACRO(double, mKp) \
	MACRO(double, mKv) \
	/* frame */ \
	MACRO(int, kinematicsFPS) \
	MACRO(int, physicsFPS) \
	/* kinematics editing */ \
	MACRO(int, blendingFrame) \
	MACRO(int, motionmatchingOffset) \
	/* motion matching cost */ \
	MACRO(std::vector<std::string>, featureBodyVelocity) \
	MACRO(std::vector<std::string>, featureBodyPosition) \
	MACRO(std::vector<int>, featureFrame) \
	/* foot contact */ \
	MACRO(std::vector<std::string>, footList) \
	MACRO(double, footContactHeight) \
	MACRO(double, footContactVelocity) \
	/* motion database list */ \
	MACRO(std::vector<std::string>, motionDataFiles) \
	/* reward components */ \
	MACRO(std::vector<std::string>, interestBodies) \
	MACRO(std::vector<std::string>, rewardBodies) \
	MACRO(std::vector<std::string>, endEffectors) \
	MACRO(std::string, head) \
	/* reward parameters */ \
	MACRO(double, torqueSoftLimit) \
	MACRO(double, speedMultiplier) \
	/* envorinment early termination */ \
	MACRO(double, rootHeightLowerLimit) \
	MACRO(double, rootHeightUpperLimit) \
	MACRO(double, rootDiffThreshold) \
	MACRO(double, rootAngleDiffThreshold) \
	/* environment step limit */ \
	MACRO(double, timeLimit) \
	/* foot phase */ \
	MACRO(double, leftSwingStart) \
	MACRO(double, leftSwingEnd) \
	MACRO(double, rightSwingStart) \
	MACRO(double, rightSwingEnd) \
	/* random pushed component */ \
	MACRO(std::vector<std::string>, pushedBody) \

namespace Parameter{
	void loadParameter();
	
	// motion matching helper function
	extern int maxFrame, featureLength;
	extern int poseLength, trajectoryLength, footContactLength;

#define PARAMHEADER(TYPE,NAME) extern TYPE NAME;
	PARAM(PARAMHEADER)
};

#endif
