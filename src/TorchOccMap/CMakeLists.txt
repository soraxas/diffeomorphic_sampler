# Define lists of files
set(
        SOURCE_FILES
)
set(
        HEADER_FILES
        ${PROJECT_SOURCE_DIR}/include/TorchOccMap.hpp
)
set(
        LINK_LIBARARIES
        torch
        # "${TORCH_LIBRARIES}"
)

find_package(Torch REQUIRED)

# Build library
add_library(torch_occ_map ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(
        torch_occ_map
        PROPERTIES
        LINKER_LANGUAGE CXX
        CXX_STANDARD 14
        )
target_link_libraries(torch_occ_map PRIVATE ${LINK_LIBARARIES})
#target_link_libraries(your-executable PUBLIC torch::Tensor)


# see url for the following two required linking
# https://github.com/pytorch/examples/blob/master/cpp/mnist/CMakeLists.txt
target_compile_features(torch_occ_map PRIVATE cxx_range_for)
# target_link_libraries(torch_occ_map PRIVATE ${TORCH_LIBRARIES})
target_include_directories(torch_occ_map PUBLIC ${TORCH_INCLUDE_DIRS} /home/soraxas/ROS/torch/libtorch/include/torch/csrc/api/include/torch)


target_include_directories(torch_occ_map PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:>  # <prefix>/include/mylib
)

# Installation setup
set_target_properties(
        torch_occ_map
        PROPERTIES
        PUBLIC_HEADER "${HEADER_FILES}"
        IMPORTED_LINK_DEPENDENT_LIBRARIES "${LINK_LIBARARIES}"
)

install(
        TARGETS torch_occ_map
        EXPORT torch_occ_map
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/include
)

# # Add all targets to the build-tree export set
# export(
#         TARGETS torch_occ_map
#         APPEND FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
# )
