cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(diffeomorphic_sampler CXX)

########################################
# Load libtorch
set(LIBTORCH_ROOT $ENV{HOME}/ROS/torch/libtorch)

if (NOT EXISTS ${LIBTORCH_ROOT})
    message( FATAL_ERROR "Folder '$LIBTORCH_ROOT' does not exists. Make sure LIBTORCH_ROOT points to libtorch's c++ library." )
endif()

# add the libtorch prefix
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_ROOT})

# include_directories(
#   ${LIBTORCH_ROOT}/include
#   ${LIBTORCH_ROOT}/include/torch/csrc/api/include
#   ${CMAKE_SOURCE_DIR}
#   )
########################################

# message("--------------")
# message(${TORCH_LIBRARIES})
# message("--------------")


add_subdirectory(src)
