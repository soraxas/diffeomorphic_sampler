
add_executable(example-app example-app.cpp)
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

find_package(Torch REQUIRED)
# target_compile_features(example-app PUBLIC cxx_range_for)
# target_link_libraries(example-app ${TORCH_LIBRARIES})


target_link_libraries(example-app torch_occ_map)
# target_link_libraries(${LIBRARY_TARGET_NAME} ${torch_occ_map_LIBRARIES})
target_include_directories(example-app PRIVATE ${PROJECT_SOURCE_DIR}/include)

find_package(Eigen3 REQUIRED)
target_link_libraries (example-app Eigen3::Eigen)

find_package(OpenMP REQUIRED)
target_link_libraries(example-app OpenMP::OpenMP_CXX)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

