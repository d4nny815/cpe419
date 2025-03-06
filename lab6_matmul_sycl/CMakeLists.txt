cmake_minimum_required(VERSION 3.4)

# Name of the project
project(sysl_lab)

# Set the compiler
set(CMAKE_CXX_COMPILER icpx)

# Get all source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Define target name
set(TARGET_NAME ${CMAKE_PROJECT_NAME})

# Set the executable
add_executable(${TARGET_NAME} ${SOURCES})

# OS-specific options and libraries
message(STATUS "Adding GCC style compiler flags")
set(COMPILE_FLAGS "-fsycl -Wall -pedantic -Werror=return-type -g")
set(LINK_FLAGS "-fsycl")

# Apply flags to the target
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")