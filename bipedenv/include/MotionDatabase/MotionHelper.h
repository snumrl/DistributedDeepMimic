#include <Eigen/Dense>
#include <algorithm>
#include <tuple>
#include <vector>
#include "MotionDatabase/MotionDB.h"
#include "MotionDatabase/MotionStatus.h"

bool isFootContact(BodyNodePtr n);
Eigen::VectorXd blendingPosition(Eigen::VectorXd fromPosition, Eigen::VectorXd toPosition, double frame, double blendingFrame);
template<typename T> void setMotionAlignment(MotionStatusLeafPtr<T> current, Eigen::Vector6d target);
template<typename T> MotionStatusPtr<T> getMotionStatusOffset(MotionStatusPtr<T> to, Eigen::VectorXd currentPosition, SkeletonPtr skel);
template<typename T> MotionStatusPtr<T> getMotionStatusFootIK(MotionStatusPtr<T> from, SkeletonPtr skel, SkeletonPtr skelcopy);
