#ifndef DRAWABLE
#define DRAWABLE

#include <memory>
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include<GL/freeglut.h>
#include <GLFW/glfw3.h>

void addFlag(int &state, int value);
void removeFlag(int &state, int value);
void flipFlag(int &state, int value);
bool isFlagSet(int state, int value);

// Drawable
class Drawable{
public:
	Drawable();
	virtual void display() = 0;

	// glut
	virtual void nextTimestep(int time);
	virtual void reshape(int width, int height);
	virtual void keyboard(unsigned char key, int x, int y);
	virtual void keyboardUp(unsigned char key, int x, int y);
	virtual void special(int key, int x, int y);
	virtual void specialUp(int key, int x, int y);
	virtual void mouse(int button, int state, int x, int y);
	virtual void motion(int x, int y);
	virtual void passiveMotion(int x, int y);
	
	// glfw
	virtual void mouse_button_callback(GLFWwindow* window, int button, int action, int mode);
	virtual void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	virtual void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	
	int width, height;
};
using DrawablePtr = std::shared_ptr<Drawable>;

// DrawableWrap

template <typename T>
class DrawableWrap : Drawable{
public:
	DrawableWrap(T object);
	
	void display() override;
private:
	T object;
};

template <typename T>
DrawableWrap<T>::DrawableWrap(T object):
		object(object){
}

template <typename T>
void DrawableWrap<T>::display(){
	displayFunction(object);
}

#endif
