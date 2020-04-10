#include<string>
#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include<GL/freeglut.h>

namespace Shader{
	void printShaderInfoLog(GLuint shader);
	void printProgramInfoLog(GLuint obj);
	GLuint createProgram(std::string vertexShaderFile, std::string fragShaderFile);
	void useShader(int on);
}
