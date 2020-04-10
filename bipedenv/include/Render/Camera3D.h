#ifndef CAMERA3D
#define CAMERA3D

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include<GL/freeglut.h>
#include<Eigen/Dense>
#include "Drawable.h"

class Camera3D : public Drawable{
public:
	Camera3D(Eigen::Vector3d origin, Eigen::Vector3d camera, Eigen::Vector3d up, double FOV);
	void display();
	void keyboard(unsigned char key, int x, int y);
	void keyboardUp(unsigned char key, int x, int y);
	void special(int key, int x, int y);
	void specialUp(int key, int x, int y);
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
	
	void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
	void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	Eigen::Vector3d origin, camera, up;
	double FOV;
	int status, Ox, Oy;
	Eigen::Quaterniond rot;

	void zoom(int dy);
	void doly(int dy);
	void translate(int dx, int dy);
	void rotate(Eigen::Quaterniond rotation);
	void rotate(int Ox, int Oy, int x, int y);
	void rotate(Eigen::Vector2d joystickAxis);
};

using Camera3DPtr = std::shared_ptr<Camera3D>;
	
#endif
