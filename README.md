# Diffeomorphic Sampler

This module is to be used as an extension to the OMPL. This requires injecting code to OMPL's CMakeLists (such that this repo is built as a sub-directory), and injecting into ompl planners' code that allocates samplers.

## Changes to CMakeLists

Required changes to `src/ompl/CMakeLists.txt`:

```cmake
##################################################################
add_subdirectory(
    $ENV{HOME}/ROS/ws_jaco-diff/src/diffeomorphic_sampler  # <---- this might need to be changed
    diff-samp
)
#find_package(catkin REQUIRED COMPONENTS diffeomorphic_state_sampler)
target_link_libraries(ompl PUBLIC diffeomorphic_state_sampler)
##################################################################
## Link to libtorch
find_package(Torch REQUIRED)
target_compile_features(ompl PRIVATE cxx_range_for)
target_link_libraries(ompl PRIVATE ${TORCH_LIBRARIES})
##################################################################
## Include necessary headers from MoveIt
set(MOVEIT_SRC_ROOT $ENV{HOME}/ROS/ws_jaco-diff/src/moveit)  # <---- this might need to be changed
target_include_directories(
        ompl PRIVATE
        ${MOVEIT_SRC_ROOT}/moveit_planners/ompl/ompl_interface/include
        ${MOVEIT_SRC_ROOT}/moveit_core/robot_model/include
        ${MOVEIT_SRC_ROOT}/moveit_core/macros/include
        ${MOVEIT_SRC_ROOT}/moveit_core/exceptions/include
        /opt/ros/melodic/include  # <---- this might need to be changed
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


