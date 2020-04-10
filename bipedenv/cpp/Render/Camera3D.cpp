#include<cmath>
#include<iostream>
#include "Render/Camera3D.h" 

// Camera3D

const int EVENT_ROTATION = 1;
const int EVENT_TRANSLATION = 2;
const int EVENT_ZOOM = 4;
const int EVENT_DOLY = 8;
const double PI = acos(-1);

Camera3D::Camera3D(Eigen::Vector3d origin, Eigen::Vector3d camera, Eigen::Vector3d up, double FOV) :
		origin(origin), camera(camera), up(up), FOV(FOV){
	status = Ox = Oy = 0;
	Eigen::Vector3d cam = (camera - origin).normalized();
	up = (up - cam.dot(up) * cam).normalized();
	rot = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d(1, 0, 0), cam);
	Eigen::Vector3d tmp = rot._transformVector(Eigen::Vector3d(0, 1, 0));
	rot = Eigen::AngleAxisd(atan2(tmp.cross(up).dot(cam), tmp.dot(up)), cam) * rot;
	printf("DOLY : x + drag (up/down)\n");
	printf("ZOOM : z + drag (up/down)\n");
}

void Camera3D::display()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(FOV, (GLfloat)width / (GLfloat)height, .1f, 1e3);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(camera[0], camera[1], camera[2], 
		origin[0], origin[1], origin[2], 
		up[0], up[1], up[2]);

	if(status){
		glPushMatrix();
		glTranslated(origin[0], origin[1], origin[2]);
		glBegin(GL_LINES);
		glColor3f(1.0, 0.0, 0.0); glVertex3d(0.0, 0.0, 0.0); glVertex3d(1.0, 0.0, 0.0); 
		glColor3f(0.0, 1.0, 0.0); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 1.0, 0.0); 
		glColor3f(0.0, 0.0, 1.0); glVertex3d(0.0, 0.0, 0.0); glVertex3d(0.0, 0.0, 1.0);
		glEnd();
		glPopMatrix();
	}
}

void Camera3D::keyboard(unsigned char key, int x, int y){
	switch(key){
		case 'x': addFlag(status, EVENT_DOLY); break;
		case 'z': addFlag(status, EVENT_ZOOM); break;
	}
}

void Camera3D::keyboardUp(unsigned char key, int x, int y){
	switch(key){
		case 'x': removeFlag(status, EVENT_DOLY); break;
		case 'z': removeFlag(status, EVENT_ZOOM); break;
	}
}

void Camera3D::special(int key, int x, int y){
}

void Camera3D::specialUp(int key, int x, int y){
}

void Camera3D::mouse(int button, int state, int x, int y){
	int flag = 0;
	switch(button){
		case GLUT_LEFT_BUTTON: flag = EVENT_ROTATION; break;
		case GLUT_RIGHT_BUTTON: flag = EVENT_TRANSLATION; break;
		default: return;
	}
	switch(state){
		case GLUT_DOWN: addFlag(status, flag); Ox = x; Oy = y; break;
		case GLUT_UP: removeFlag(status, flag); break;
	}
}

void Camera3D::zoom(int dy){ 
	FOV += 0.05 * dy;
	if( FOV < 10 ) FOV = 10;
	if( FOV > 130 ) FOV = 130;
}

void Camera3D::doly(int dy){
	camera += (camera - origin) * (0.01f * dy);
}

void Camera3D::translate(int dx, int dy){
	Eigen::Vector3d dv = (origin - camera).normalized() * (dy * FOV / 1000) + 
		up.cross(origin - camera).normalized() * (dx * FOV / 1000);
	dv = Eigen::Vector3d(dv[0], 0, dv[2]); //.normalized() * t;
	origin += dv; camera += dv;
}

void Camera3D::rotate(Eigen::Quaterniond rotation){
	up = rotation._transformVector(up);
	camera = rotation._transformVector(camera - origin) + origin;
	rot = rotation * rot;
}

void Camera3D::rotate(int Ox, int Oy, int x, int y){
	const double R = 400.0;
	auto proj = [](Eigen::Vector3d vec){
		if(vec.norm() >= 1) vec.normalize();
		else vec[0] = -sqrt(1 - vec.dot(vec));
		return vec;
	};
	Eigen::Vector3d start = proj(Eigen::Vector3d(0, Oy/R, Ox/R));
	Eigen::Vector3d end = proj(Eigen::Vector3d(0, y/R, x/R));
	start = rot._transformVector(start);
	end = rot._transformVector(end);
	Eigen::Quaterniond move = Eigen::Quaterniond::FromTwoVectors(end, start);
	rotate(move * move);
}

void Camera3D::rotate(Eigen::Vector2d joystickAxis){
	rotate(Eigen::Quaterniond(
		Eigen::AngleAxisd(joystickAxis[0] * 0.1, Eigen::Vector3d(0, 1, 0))));

	Eigen::Vector3d dir = (origin-camera).normalized();
	Eigen::Vector3d left = dir.cross(up);
	double angle = atan2(dir.cross(Eigen::Vector3d(0, 1, 0)).dot(left), 
		dir.dot(Eigen::Vector3d(0, 1, 0)));

	double dx = -joystickAxis[1] * 0.1;
	if(angle + dx > PI - 0.1) dx = PI - 0.1 - angle;
	if(angle + dx < 0.1) dx = 0.1 - angle;
	
	rotate(Eigen::Quaterniond(
		Eigen::AngleAxisd(-dx, left)));
}

void Camera3D::motion(int x, int y){
	if     (isFlagSet(status, EVENT_ZOOM)) zoom(y-Oy);
	else if(isFlagSet(status, EVENT_DOLY)) doly(y-Oy);
	else if(isFlagSet(status, EVENT_TRANSLATION)) translate(x-Ox, y-Oy);
	else if(isFlagSet(status, EVENT_ROTATION))
		rotate(Ox-width/2, Oy-height/2, x-width/2, y-height/2);
	Ox = x; Oy = y;
}

void Camera3D::mouse_button_callback(GLFWwindow* window, int button, int action, int mode){
	int flag = 0;
	double x, y; glfwGetCursorPos(window, &x, &y);
	switch(button){
		case GLFW_MOUSE_BUTTON_LEFT: flag = EVENT_ROTATION; break;
		case GLFW_MOUSE_BUTTON_RIGHT: flag = EVENT_TRANSLATION; break;
		default: return;
	}
	switch(action){
		case GLFW_PRESS: addFlag(status, flag); Ox = x; Oy = y; break;
		case GLFW_RELEASE: removeFlag(status, flag); break;
	}
}

void Camera3D::cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	motion(xpos, ypos);
}

void Camera3D::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
	switch(action){
		case GLFW_PRESS: switch(key){
			case GLFW_KEY_X: flipFlag(status, EVENT_DOLY); break;
			case GLFW_KEY_Z: flipFlag(status, EVENT_ZOOM); break;
		}
	}
}