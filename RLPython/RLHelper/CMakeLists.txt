
cmake_minimum_required(VERSION 3.5.1)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

get_filename_component(project_name RLHelper NAME)

add_compile_options(-fPIC -fopenmp)
add_compile_options(-std=c++1z)
add_compile_options(-Wall)
add_compile_options(-lm)
#add_compile_options(-pg)
#add_compile_options(-O2)

set(required_components collision-bullet utils-urdf gui)
set(required_libraries dart dart-collision-bullet dart-utils-urdf dart-gui)
find_package(DART 6.6.0 REQUIRED COMPONENTS ${required_components} CONFIG)
find_package(Boost COMPONENTS python3 numpy3)
find_package(PythonLibs REQUIRED)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})

file(GLOB srcs *.cpp)
add_library(${project_name} SHARED ${srcs})
target_link_libraries(${project_name} PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
add_executable(a.out ${srcs} ${CUDA_OBJ})
target_link_libraries(a.out PUBLIC ${required_libraries} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
