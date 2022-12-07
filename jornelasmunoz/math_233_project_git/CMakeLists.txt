cmake_minimum_required(VERSION 3.23)
project(math_233_project)

set(CMAKE_CXX_STANDARD 14)

add_executable(math_233_project
        main.cpp
        mnist_utils.cpp
        mnist_utils.h
        MURA.cpp
        MURA.h
        FullMatrix.h
        FullMatrix.cpp
        /usr/local/include/fftw3.h)

