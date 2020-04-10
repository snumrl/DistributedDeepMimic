cmake_minimum_required(VERSION 3.5.1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

get_filename_component(project_name biped NAME)

set(CMAKE_MODULE_PATH "${CMAKE_HOME_DIRECTORY}/cmake_module")
set(required_components collision-bullet utils-urdf gui external-odelcpsolver)
set(required_libraries dart dart-collision-bullet dart-utils-urdf dart-gui GL GLU GLEW glut glfw tinyxml dart-external-odelcpsolver)

add_compile_options(-fPIC -faligned-new)
add_compile_options(-fopenmp)
add_compile_options(-march=native)
add_compile_options(-std=c++1z)
add_compile_options(-Wall)
add_compile_options(-lm)
add_compile_options(-pg)

find_package(DART 6.9.2 REQUIRED COMPONENTS ${required_components} CONFIG)
find_package(Boost COMPONENTS python3 numpy3)
find_package(PythonLibs REQUIRED)
find_package(OpenMP)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories("${CMAKE_HOME_DIRECTORY}/include")

file(GLOB env_srcs cpp/*.cpp cpp/Helper/*.cpp cpp/Agents/*.cpp cpp/Render/*.cpp cpp/MotionDatabase/*.cpp cpp/Environment/*.cpp)
add_library(${project_name} SHARED ${env_srcs})
target_link_libraries(${project_name} PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
add_executable(a.out ${env_srcs})
target_link_libraries(a.out PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} ${OpenMP_CXX_FLAGS})
