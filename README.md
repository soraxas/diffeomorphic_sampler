# Diffeomorphic Sampler

This module is to be used as an extension to the OMPL. This requires injecting code to OMPL's CMakeLists (such that this repo is built as a sub-directory), and injecting into ompl planners' code that allocates samplers.

**IMPORTANT:** Note that since this sampler depends on `MoveitIt`'s robot model and `OMPL`'s state space, and yet this module will need to be inject within ompl (hence circular dependency: MoveIt -> OMPL -> diffeomorphic-sampler -> MoveIt), this will requires some tricky to be compile successfully.

1. First compile the entire ROS workspace as usual, *without* any of the following changes. (If you had already modified `ompl` source, you can use git to revert to one of the official commit that contains none of these changes)
    ```sh
    # e.g.
    cd ws_sbp
    cd src/ompl
    git checkout origin/main
    cd ../..
    catkin build
    ```
2. Then, both MoveIt and OMPL inside the workspace (e.g. `ws_sbp`) should now be built. Modifies the changes specified below to OMPL, make sure the variables (e.g. `MOVEIT_SRC_ROOT` etc.) points to the correct path.
3. After applying the changes to OMPL src, rerun `catkin build` in the workspace. This works because it can link to the older version of moveit (older, but not modified).

## Changes to CMakeLists

Required changes to `src/ompl/CMakeLists.txt`:

```cmake
##################################################################
## NECESSARY DIR SETTINGS by user
set(DIFF_SAMP_DIR $ENV{HOME}/research/diffeomorphic_sampler)
#set(MOVEIT_SRC_ROOT $ENV{HOME}/ROS/ws_jaco-diff/src/moveit)
set(MOVEIT_SRC_ROOT ${CMAKE_SOURCE_DIR}/../moveit)
# typical var catkin_DIR=/opt/ros/noetic/share/catkin/cmake
# we will walk backward to find system include path
set(CATKIN_SYSTEM_INCLUDE "${catkin_DIR}/../../../include")
##################################################################
function(ensure_path_exists _dir)
  if(NOT IS_DIRECTORY "${_dir}" OR NOT EXISTS "${_dir}")
    message(FATAL_ERROR "The given directory '${_dir}' does not exists!")
  endif()
endfunction(ensure_path_exists)
## Ensure path exists, sanity check!
ensure_path_exists("${DIFF_SAMP_DIR}")
ensure_path_exists("${MOVEIT_SRC_ROOT}")
ensure_path_exists("${CATKIN_SYSTEM_INCLUDE}")
##################################################################
## add the actual source for diff sampler
add_subdirectory(${DIFF_SAMP_DIR} diff-samp)
#find_package(catkin REQUIRED COMPONENTS diffeomorphic_state_sampler)
target_link_libraries(ompl PUBLIC diffeomorphic_state_sampler)
##################################################################
## Link to libtorch
find_package(Torch REQUIRED)
target_compile_features(ompl PRIVATE cxx_range_for)
target_link_libraries(ompl PRIVATE ${TORCH_LIBRARIES})
##################################################################
## Include necessary headers from MoveIt
target_include_directories(
        ompl PRIVATE
        ${MOVEIT_SRC_ROOT}/moveit_planners/ompl/ompl_interface/include
        ${MOVEIT_SRC_ROOT}/moveit_core/robot_model/include
        ${MOVEIT_SRC_ROOT}/moveit_core/macros/include
        ${MOVEIT_SRC_ROOT}/moveit_core/exceptions/include
        #/opt/ros/melodic/include
        ${CATKIN_SYSTEM_INCLUDE}
)
##################################################################
## Link to MoveIt interface for robot model
#find_package(moveit_planners_ompl REQUIRED)
#target_link_libraries(ompl PRIVATE moveit_ompl_interface)
find_library(
        MOVEIT_OMPL_INTERFACE_LIB moveit_ompl_interface
        HINTS ${CATKIN_DEVEL_PREFIX}/lib)
target_link_libraries(ompl PRIVATE ${MOVEIT_OMPL_INTERFACE_LIB})
##################################################################
```

Other instances of `target_link_libraries` might also need to be updated from

```cmake
target_link_libraries(ompl foobar)
```

to

```cmake
target_link_libraries(ompl PUBLIC foobar)
```

## Changes to planner's Header (`.hpp`)

The following are used to control the planner's behavior within Rviz.

```cpp
bool diff__use_diff{true};
double diff__epsilon{0.5};
int diff__num_drift{2};
int diff__rand_batch_sample_size{400};
double diff__radius_of_joint{0.05};
```

## Changes to planner's transnational unit (`.cpp`)

At the beginning of the file:

```cpp
/////////////////////////////////////////////
#include "cppm.hpp"
#include "soraxas_cpp_toolbox/globals.h"
#include "soraxas_cpp_toolbox/stats.h"
/////////////////////////////////////////////
#define DIFFEOMORPHIC_SAMPLER_FOR_MOVEIT yes
#define CSPACE_NUM_DIM 6
#define WORLDSPACE_NUM_DIM 3
#include "DiffeomorphicStateSampler.hpp"
using DiffeomorphicSamplerType = DiffeomorphicStateSampler<ompl_interface::JointModelStateSpace, CSPACE_NUM_DIM>;
/////////////////////////////////////////////
```

Inside the constructor of the planner:

```cpp
ompl::geometric::MyPlanner::MyPlanner(const base::SpaceInformationPtr &si, ...)
{
    /////////////////////////////////////////////
    Planner::declareParam_lambda<bool>(
        "use_diff",
        [this](bool in) {
            std::cout << "----- setting use_diff as " << in << "-----" << std::endl
            diff__use_diff = in
        },
        [this](){ return diff__use_diff; }, "0,1");
    Planner::declareParam_lambda<int>(
        "diff_batch_size", [this](int in) { diff__rand_batch_sample_size = in; },
        [this]() { return diff__rand_batch_sample_size; }, "50:1:1000");
    Planner::declareParam_lambda<double>(
        "diff_epsilon", [this](double in) { diff__epsilon = in; }, [this]() { return diff__epsilon; }, "0.5:0.01:2.");
    Planner::declareParam_lambda<int>(
        "num_diff", [this](int in) { diff__num_drift = in; }, [this]() { return diff__num_drift; }, "2:1:10");
    Planner::declareParam_lambda<double>(
        "radius_of_joint", [this](double in) { diff__radius_of_joint = in; },
        [this]() { return diff__radius_of_joint; }, "0.05,0.01:2.");
    /////////////////////////////////////////////
    ...
    ...
}
```

At the end of the `solve(...)` function:

```cpp
ompl::geometric::MyPlanner::solve(...)
{
    ...
    while (ptc)
    {
        ...
    }
    ...
    timer.finish();
    std::static_pointer_cast<DiffeomorphicSamplerType>(sampler_)->finish_sampling();
    sxs::g::print_all_stored_stats();
    ...
}
```

And within the `allocSampler()` function to allocate the actual sampler

```cpp
void ompl::geometric::MyPlanner::allocSampler()
{
    /////////////////////////////////////////////
    sxs::g::storage::print_stored_info();
    sxs::g::storage::clear();
    sxs::g::storage::store<std::thread::id>("thread_id", std::this_thread::get_id());
    sxs::g::storage::store("si", si_);

    // replace the sampler to the diffeomorphic one
    auto diff_sampler = std::make_shared<DiffeomorphicSamplerType>(
        si_->getStateSpace().get(), diff__rand_batch_sample_size, diff__epsilon, diff__num_drift);

    diff_sampler->use_diff = diff__use_diff;
    diff_sampler->start_sampling();
    sampler_ = diff_sampler;

    return;

    ...
}
```


