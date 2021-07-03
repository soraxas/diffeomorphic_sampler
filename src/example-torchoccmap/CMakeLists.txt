
add_executable(example-app-occmap example-app.cpp)
set_property(TARGET example-app-occmap PROPERTY CXX_STANDARD 14)

target_link_libraries(example-app-occmap PRIVATE torch_occ_map)
target_include_directories(example-app-occmap PRIVATE ${PROJECT_SOURCE_DIR}/include)

##################################################################
## Link to libtorch
find_package(Torch REQUIRED)

target_compile_features(example-app-occmap PRIVATE cxx_range_for)
target_link_libraries(example-app-occmap PRIVATE ${TORCH_LIBRARIES})
##################################################################


find_package(Eigen3 REQUIRED)
target_link_libraries (example-app-occmap PRIVATE Eigen3::Eigen)

find_package(OpenMP REQUIRED)
target_link_libraries(example-app-occmap PRIVATE OpenMP::OpenMP_CXX)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app-occmap
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app-occmap>)
endif (MSVC)

