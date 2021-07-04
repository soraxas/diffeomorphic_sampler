# Define lists of files
set(
        SOURCE_FILES
)
set(
        HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/DiffeomorphicStateSampler.hpp
)
set(
        LINK_LIBARARIES
        torch_occ_map
)


# find_package(Torch REQUIRED CONFIG)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# Build library
add_library(diffeomorphic_state_sampler ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(
        diffeomorphic_state_sampler
        PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 17
        )
target_link_libraries(diffeomorphic_state_sampler PRIVATE ${LINK_LIBARARIES})


target_include_directories(diffeomorphic_state_sampler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:>  # <prefix>/include/mylib
)


target_link_libraries(diffeomorphic_state_sampler PUBLIC torch_occ_map)
target_include_directories(diffeomorphic_state_sampler PUBLIC ${PROJECT_SOURCE_DIR}/include)

##################################################################
## Link to libtorch
find_package(Torch REQUIRED)

target_compile_features(diffeomorphic_state_sampler PRIVATE cxx_range_for)
target_link_libraries(diffeomorphic_state_sampler PRIVATE ${TORCH_LIBRARIES})
##################################################################

find_package(Eigen3 REQUIRED)
target_link_libraries (diffeomorphic_state_sampler PUBLIC Eigen3::Eigen)

find_package(OpenMP REQUIRED)
target_link_libraries(diffeomorphic_state_sampler PUBLIC OpenMP::OpenMP_CXX)



##################################################################
## Link to OMPL
find_package(ompl REQUIRED)
# find_package(catkin REQUIRED COMPONENTS
#              ompl
#              )
target_include_directories(
        diffeomorphic_state_sampler PUBLIC
        ${OMPL_INCLUDE_DIRS}
)
# target_link_libraries(diffeomorphic_state_sampler ${ompl_LIBRARIES})
# target_link_libraries(diffeomorphic_state_sampler ${catkin_LIBRARIES})
# target_include_directories(diffeomorphic_state_sampler RPIVATE ${OMPL_INCLUDE_DIRS})
# target_link_libraries(diffeomorphic_state_sampler PRIVATE ompl)
##################################################################
## Link to MoveIt
# find_package(
#     catkin REQUIRED COMPONENTS
#    moveit_ros_planning_interface
# )
# find_package(
#     moveit_ros_planning_interface REQUIRED
# )
# find_library(
#         MOVEIT_OMPL_INTERFACE_LIB moveit_ompl_interface
#         HINTS /home/soraxas/ROS/ws_jaco-diff/devel_debug/lib
#         )
# target_link_libraries(diffeomorphic_state_sampler PRIVATE ${MOVEIT_OMPL_INTERFACE_LIB})
##################################################################
# find_package(
#     moveit_ros_planning_interface REQUIRED
# )
# target_include_directories(
#         diffeomorphic_state_sampler PRIVATE
#         /home/soraxas/ROS/ws_jaco-diff/src/moveit/moveit_planners/ompl/ompl_interface/include
#         /home/soraxas/ROS/ws_jaco-diff/src/moveit/moveit_core/robot_model/include
#         /home/soraxas/ROS/ws_jaco-diff/src/moveit/moveit_core/macros/include
#         /home/soraxas/ROS/ws_jaco-diff/src/moveit/moveit_core/exceptions/include
#         )
##################################################################


# include(FetchContent)
# # ==================================================
# set (EXT_EIGENRAND "ext_eigenrand")
# FetchContent_Declare (
#   ${EXT_EIGENRAND}

#   PREFIX         ${EXT_EIGENRAND}
#   GIT_REPOSITORY https://github.com/bab2min/EigenRand
#   GIT_TAG        111e6888bda0302348abc301096a24e07d20d354 #v0.3.4
#   GIT_SHALLOW    ON
#   SOURCE_SUBDIR  cmake

#   BUILD_ALWAYS   OFF
#   INSTALL_DIR    ${CMAKE_CURRENT_BINARY_DIR}/ext/${EXT_EIGENRAND}

#   CMAKE_ARGS
#     -DBUILD_TESTS=OFF
#   CMAKE_CACHE_ARGS
#     -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>

# # BUILD_COMMAND     ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target INSTALL
# )
# # ==================================================





# Installation setup
set_target_properties(
        diffeomorphic_state_sampler
        PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        IMPORTED_LINK_DEPENDENT_LIBRARIES "${LINK_LIBARARIES}"
)

install(
        TARGETS diffeomorphic_state_sampler
        EXPORT diffeomorphic_state_sampler
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

