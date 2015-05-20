# CMakefile for landmarker : testing landmarker detection of dlib
# (http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html)

cmake_minimum_required(VERSION 2.8.4)

PROJECT(landmarks)

# need OpenCV :
#--------------
find_package(OpenCV REQUIRED )

# using c++11 :
#--------------
add_definitions(-std=c++11)

# call libraries :
#-----------------
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
# the following line depends on where you installed dlib :
# include( --path-to-dlib-repository--/dlib/cmake)
include(/home/alexis/src/dlib-18.15/dlib/cmake)

# executables :
#--------------
ADD_EXECUTABLE(head_pos src/test_pos.cpp)
ADD_EXECUTABLE(eye_pos src/eye_pos.cpp)
TARGET_LINK_LIBRARIES(head_pos dlib ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(eye_pos dlib ${OpenCV_LIBS})

# installation :
#---------------
install(TARGETS head_pos RUNTIME DESTINATION bin)
file(GLOB_RECURSE data share/*)
install(FILES ${data} DESTINATION bin)
