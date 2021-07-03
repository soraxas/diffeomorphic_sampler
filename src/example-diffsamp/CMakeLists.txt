
add_executable(example-app-diffsamp example-app.cpp)
set_property(TARGET example-app-diffsamp PROPERTY CXX_STANDARD 17)

# target_link_libraries(example-app-diffsamp torch_occ_map)
# # target_link_libraries(${LIBRARY_TARGET_NAME} ${torch_occ_map_LIBRARIES})
# target_include_directories(example-app-diffsamp PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(example-app-diffsamp PUBLIC diffeomorphic_state_sampler)

##################################################################
## Link to libtorch
find_package(Torch REQUIRED)

target_compile_features(example-app-diffsamp PRIVATE cxx_range_for)
target_link_libraries(example-app-diffsamp PRIVATE ${TORCH_LIBRARIES})
##################################################################


# # The following code block is suggested to be used on Windows.
# # According to https://github.com/pytorch/pytorch/issues/25457,
# # the DLLs need to be copied to avoid memory errors.
# if (MSVC)
#   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
#   add_custom_command(TARGET example-app-diffsamp
#                      POST_BUILD
#                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
#                      ${TORCH_DLLS}
#                      $<TARGET_FILE_DIR:example-app-diffsamp>)
# endif (MSVC)

