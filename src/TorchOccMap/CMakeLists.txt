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


find_package(Torch REQUIRED CONFIG)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


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
target_compile_features(torch_occ_map PRIVATE PUBLIC cxx_range_for)
target_link_libraries(torch_occ_map PUBLIC ${TORCH_LIBRARIES})


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
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}
        PUBLIC_HEADER DESTINATION ${INSTALL_INCLUDE_DIR}
        INCLUDES DESTINATION ${INSTALL_INCLUDE_DIR}
)

# # Add all targets to the build-tree export set
# export(
#         TARGETS torch_occ_map
#         APPEND FILE "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
# )
