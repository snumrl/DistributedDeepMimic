#include "Render/Drawable.h"

void addFlag(int &state, int value){ state |= value; }
void removeFlag(int &state, int value){ state &= -1-value; }
void flipFlag(int &state, int value){ state ^= value; }
bool isFlagSet(int state, int value){ return state & value; }

// glut
Drawable::Drawable(){}
void Drawable::nextTimestep(int time){}
void Drawable::reshape(int width, int height){
	this->width = width; this->height = height;
}
void Drawable::keyboard(unsigned char key, int x, int y){}
void Drawable::keyboardUp(unsigned char key, int x, int y){}
void Drawable::special(int key, int x, int y){}
void Drawable::specialUp(int key, int x, int y){}
void Drawable::mouse(int button, int state, int x, int y){}
void Drawable::motion(int x, int y){}
void Drawable::passiveMotion(int x, int y){}

// glfw
void Drawable::mouse_button_callback(GLFWwindow* window, int button, int action, int mode){}
void Drawable::cursor_position_callback(GLFWwindow* window, double xpos, double ypos){}
void Drawable::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){}