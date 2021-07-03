//
// Created by soraxas on 4/7/21.
//
#pragma onceb

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateValidityChecker.h>
#include <ompl/util/Exception.h>

#include <EigenRand/EigenRand>

/////////////////////////////////////////////////////
#ifndef CSPACE_NUM_DIM
#define CSPACE_NUM_DIM 2
#endif

#ifndef WORLDSPACE_NUM_DIM
#define WORLDSPACE_NUM_DIM 2
#endif

using EigenCSpacePt           = Eigen::Array<double, 1, CSPACE_NUM_DIM>;
using EigenCSpacePtVector     = Eigen::Matrix<double, CSPACE_NUM_DIM, 1>;
using EigenWorldSpacePtVector = Eigen::Matrix<double, WORLDSPACE_NUM_DIM, 1>;
/////////////////////////////////////////////////////

#ifdef DIFFEOMORPHIC_SAMPLER_USE_HILBERT_MAP
#include "HMapConstructor.h"
#endif

#include "cppm.hpp"
#include "soraxas_cpp_toolbox/main.h"
//#define USE_DEBUG_LOG
#include "soraxas_cpp_toolbox/debug_logger.h"
#include "soraxas_cpp_toolbox/external/csv.hpp"
#include "soraxas_cpp_toolbox/globals.h"
#include "soraxas_cpp_toolbox/main.h"

#ifdef DIFFEOMORPHIC_SAMPLER_FOR_MOVEIT
/////////////////////////////////////////////
#include <moveit/ompl_interface/parameterization/joint_space/joint_model_state_space.h>
#include <moveit/ompl_interface/parameterization/joint_space/joint_model_state_space_factory.h>
/////////////////////////////////////////////
#endif

//#define SXS_LOG_TO_CSV
//#define SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
//#define COLLECT_ANGLES_DISTRIBUTION
//#define SXS_DIFF_SAMP_LOG_STATS

#include "soraxas_cpp_toolbox/external/concurrentqueue/concurrentqueue.h"

namespace ob = ompl::base;
// namespace og = ompl::geometric;

#include "TorchOccMap.hpp"

//#define CSPACE_NUM_DIM Eigen::Dynamic
//#define CSPACE_NUM_DIM 2

template <typename StateSpaceType, size_t NumDim>
class DiffeomorphicStateSampler_Base : public ob::StateSampler {
  using EigenCSpacePt       = Eigen::Array<double, 1, NumDim>;
  using EigenCSpacePtVector = Eigen::Matrix<double, NumDim, 1>;

 public:
  using MatrixXdRowMajor =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using MatrixXfRowMajor =
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ArrayXdRowMajor =
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  using SampledMatRowMajor =
      Eigen::Array<double, Eigen::Dynamic, NumDim, Eigen::RowMajor>;

  DiffeomorphicStateSampler_Base(
      const ob::StateSpace* space,
      //      std::shared_ptr<HMDiff> hilbert_map,
      //                                 TorchOccMapManager::TorchOccMapPtr
      //                                 occ_map,
      int rand_batch_sample_size = 50, double epsilon = 2, uint num_drift = 2)
      : StateSampler(space),
        rand_batch_sample_size(rand_batch_sample_size),
        //        m_hilbert_map(std::move(hilbert_map)),
        //        m_occ_map(std::move(occ_map)),
        //        m_occ_map(),
        //        m_occ_map(std::make_shared<TorchOccMap>(
        //            "/home/soraxas/Downloads/diff_nn/occmap.pt")),
        epsilon(epsilon),
        num_drift(num_drift),
        num_dimensions(space_->getDimension()),
        bounds_high(std::vector<double>()),
        bounds_low(std::vector<double>()) {
    if (num_dimensions != NumDim) {
      std::cout << "Given state-space dim " << num_dimensions
                << " is different than the defined dimension " << NumDim
                << std::endl;
      assert(num_dimensions == NumDim);
    }

    //    ////////////////////////////////////////
    //    ///// Setup state space
    //    std::string space_type;
    //    space_type = "RealVectorStateSpace";
    //    space_type = "JointModelStateSpace";

    casted_space_ptr = dynamic_cast<const StateSpaceType*>(space_);
    assert(("Bad cast of a state-space ptr.", casted_space_ptr != nullptr));
  }

  void init() {
    // Derived class will override this function
    _retrieve_dimension_bounds();

    assert(bounds_high.size() == NumDim);
    assert(bounds_low.size() == NumDim);

    rand_urng = Eigen::Rand::Vmt19937_64{std::random_device{}()};
    rand_sampled_batch_cur_row = rand_batch_sample_size;  // we starts with
                                                          // an empty batch
    DEBUG_LOG(bounds_low);
    DEBUG_LOG(bounds_high);
    bounds_lower = EigenCSpacePt::Map(bounds_low.data(), bounds_low.size());
    bounds_upper = EigenCSpacePt::Map(bounds_high.data(), bounds_high.size());
    bounds_diff  = bounds_upper - bounds_lower;
#ifndef DIFFEOMORPHIC_SAMPLER_FOR_MOVEIT
    // init vector container
    rand_sampled_batch_as_vec =
        std::vector<EigenCSpacePtVector>(rand_batch_sample_size);
#endif
  }

  virtual void _retrieve_dimension_bounds() = 0;

  const StateSpaceType* casted_space_ptr;

  // batch size for the random number
  size_t rand_batch_sample_size;
  // random number generator
  Eigen::Rand::Vmt19937_64 rand_urng;
  // the sampled batch
  SampledMatRowMajor rand_sampled_batch;
  std::vector<EigenCSpacePtVector> rand_sampled_batch_as_vec;
  // the current row index of the sampled batch
  size_t rand_sampled_batch_cur_row{};
  // random number generate as uniform random
  Eigen::Rand::UniformRealGen<double> rand_uni_gen;

  bool use_diff = true;

  ///////////////////////////////////////////////////////////////////////

  //#define PLOT_SAMPLED_PTS

  void draw_sample_bucket() {
    //    rand_sampled_batch =
    //        Eigen::ArrayXXd::Random(rand_batch_sample_size, num_dimensions);
    rand_sampled_batch = rand_uni_gen.generate<SampledMatRowMajor>(
        rand_batch_sample_size, num_dimensions, rand_urng);
    // 0-1  =>  0-diff  =>  lower-upper
    rand_sampled_batch =
        (rand_sampled_batch.rowwise() * bounds_diff).rowwise() + bounds_lower;
    rand_sampled_batch_cur_row = 0;
#ifdef PLOT_SAMPLED_PTS
    for (int i = 0; i < rand_sampled_batch.rows(); ++i) {
      sampled_ori.emplace_back(rand_sampled_batch.row(i));
    }
#endif
    drift_states();
#ifdef PLOT_SAMPLED_PTS
    for (int i = 0; i < rand_sampled_batch.rows(); ++i) {
      sampled_diff.emplace_back(rand_sampled_batch.row(i));
    }
#endif
  }

  virtual void drift_states() {
    DEBUG_LOG("DRIFTED=======================");
#ifdef DIFFEOMORPHIC_SAMPLER_FOR_MOVEIT
    throw std::runtime_error("not implemented");
#else
/*
    for (uint i = 0; i < num_drift; ++i) {
      // map the eigen matrix to the vector
      SampledMatRowMajor::Map(rand_sampled_batch_as_vec.data()->data(),
                              rand_batch_sample_size, num_dimensions) =
          rand_sampled_batch;
      // compute gradient at the specified location
      auto _occ_grad = m_hilbert_map->m_map->occupancy_and_gradient_parallel(
          rand_sampled_batch_as_vec);
      auto _query_gradient = _occ_grad.second;

      Eigen::Map<SampledMatRowMajor> grad(
          _query_gradient[0].data(), _query_gradient.size(), num_dimensions);
      // performs the actual update
      rand_sampled_batch -= epsilon * grad;
    }
  */
  throw std::runtime_error("not implemented");
#endif
  };

  sxs::Stats& m_stats = sxs::g::get_stats();

  void sampleUniform(ob::State* state) override {
    //    sxs::g::stats.of<int>("samp") += 1;

    bool morphed;
    if (this->use_diff) {
      morphed = sampleUniform_Diffeomorphic(state);
    } else {
      sampleUniform_PurelyUniformStateSampler(state);
      morphed = false;
    }

#ifdef SXS_DIFF_SAMP_LOG_STATS
    auto si_ = sxs::g::storage::get<ob::SpaceInformationPtr>("si");
    state->as<ompl_interface::JointModelStateSpace::StateType>()
        ->clearKnownInformation();
    bool is_valid = si_->isValid(state);
    m_stats.of<long>("total_sample") += 1;
    m_stats.of<long>("total_valid") += is_valid;
    m_stats.of<double>("total_valid_pct") =
        double(m_stats.of<long>("total_valid")) /
        double(m_stats.of<long>("total_sample"));

    m_stats.of<long>("total_valid_morphed") += 0;
    m_stats.of<long>("total_sample_morphed") += 0;
    m_stats.of<long>("total_valid_uniform") += 0;
    m_stats.of<long>("total_sample_uniform") += 0;
    if (morphed) {
      m_stats.of<long>("total_valid_morphed") += is_valid;
      m_stats.of<long>("total_sample_morphed") += 1;
    } else {
      m_stats.of<long>("total_valid_uniform") += is_valid;
      m_stats.of<long>("total_sample_uniform") += 1;
    }

    m_stats.of<double>("total_valid_morphed_pct") =
        double(m_stats.of<long>("total_valid_morphed")) /
        double(m_stats.of<long>("total_sample_morphed"));

    m_stats.of<double>("total_valid_uniform_pct") =
        double(m_stats.of<long>("total_valid_uniform")) /
        double(m_stats.of<long>("total_sample_uniform"));
#endif

    //    auto& timer = sxs::g::get<cppm::pm_timer>("timer");
    //    sxs::g::stats.format_item(timer);
    //    timer.update();
  }

  void sampleUniform_PurelyUniformStateSampler(ob::State* state) {
    const unsigned int dim = space_->getDimension();

    auto* rstate = state->as<typename StateSpaceType::StateType>();
    for (unsigned int i = 0; i < dim; ++i)
      rstate->values[i] = rng_.uniformReal(bounds_low[i], bounds_high[i]);
  }

  virtual bool sampleUniform_Diffeomorphic(ob::State* state) {
    auto* rstate          = state->as<typename StateSpaceType::StateType>();
    double* rstate_values = rstate->values;

    if (rand_sampled_batch_cur_row >= rand_batch_sample_size)
      draw_sample_bucket();

    /* assign rstate_values with pointer (the matrix NEEDS to be row-major) */
    // get raw pointer from the matrix
    double* from = &rand_sampled_batch(rand_sampled_batch_cur_row++, 0);
    std::copy(from, from + num_dimensions, rstate_values);
    return true;
  }

  void sampleUniformNear(ob::State*, const ob::State*, const double) override {
    throw ompl::Exception("not implemented");
  }

  void sampleGaussian(ob::State*, const ob::State*, const double) override {
    throw ompl::Exception("not implemented");
  }

  double epsilon;
  uint num_drift;

 protected:
  //  std::shared_ptr<HMDiff> m_hilbert_map;
  //  TorchOccMapManager::TorchOccMapPtr m_occ_map;
  unsigned int num_dimensions;
  ompl::RNG rng_;
  /** \brief The sampler to build upon */

  std::vector<double> bounds_high;
  std::vector<double> bounds_low;

  EigenCSpacePt bounds_lower;
  EigenCSpacePt bounds_upper;
  EigenCSpacePt bounds_diff;
};

/**
 * The abstract base class for the diffeomorphic sampler
 *
 * @tparam StateSpaceType - The state space type to be specialised on (e.g.
 *         Real Vector or MoveIt Joint Space)
 * @tparam NumDim - The dimensionality of the space. TODO: make this generic
 */
template <typename StateSpaceType, size_t NumDim>
class DiffeomorphicStateSampler
    : public DiffeomorphicStateSampler_Base<StateSpaceType, NumDim> {};

/**
 * Specialisation for `RealVectorStateSpace`
 *
 * @tparam NumDim
 */
template <size_t NumDim>
class DiffeomorphicStateSampler<ob::RealVectorStateSpace, NumDim>
    : public DiffeomorphicStateSampler_Base<ob::RealVectorStateSpace, NumDim> {
 public:
  // call base constructor
  template <typename... Args>
  explicit DiffeomorphicStateSampler(Args... args)
      : DiffeomorphicStateSampler_Base<ob::RealVectorStateSpace, NumDim>(
            args...) {
    this->init();
  }

  void _retrieve_dimension_bounds() override {
    auto bounds = this->casted_space_ptr->getBounds();
    this->bounds_high.insert(this->bounds_high.end(), bounds.high.begin(),
                             bounds.high.end());
    this->bounds_low.insert(this->bounds_low.end(), bounds.low.begin(),
                            bounds.low.end());
  }
};

#ifdef DIFFEOMORPHIC_SAMPLER_FOR_MOVEIT
/**
 * Specialisation for `JointModelStateSpace`
 */
template <size_t NumDim>
class DiffeomorphicStateSampler<ompl_interface::JointModelStateSpace, NumDim>
    : public DiffeomorphicStateSampler_Base<
          ompl_interface::JointModelStateSpace, NumDim> {
 public:
  using SampledMatRowMajor = typename DiffeomorphicStateSampler_Base<
      ompl_interface ::JointModelStateSpace, NumDim>::SampledMatRowMajor;
  using MatrixXdRowMajor = typename DiffeomorphicStateSampler_Base<
      ompl_interface ::JointModelStateSpace, NumDim>::MatrixXdRowMajor;
  using MatrixXfRowMajor = typename DiffeomorphicStateSampler_Base<
      ompl_interface ::JointModelStateSpace, NumDim>::MatrixXfRowMajor;

  using ArrayXdRowMajor = typename DiffeomorphicStateSampler_Base<
      ompl_interface ::JointModelStateSpace, NumDim>::ArrayXdRowMajor;

  // call base constructor
  template <typename... Args>
  DiffeomorphicStateSampler(Args... args)
      : DiffeomorphicStateSampler_Base<ompl_interface ::JointModelStateSpace,
                                       NumDim>(args...),
        //        m_thread_num(4),
        //        m_thread_pool(m_thread_num),
        m_finish_sampling(false),
        m_difted_samples_Q(200, omp_get_max_threads(), omp_get_max_threads()) {
    this->init();
  }

  ~DiffeomorphicStateSampler() {
    if (background_sampling_th) {
      background_sampling_th->join();
      background_sampling_th.reset();
    }
  }

  void _retrieve_dimension_bounds() override {
    for (auto&& joint_bound : this->casted_space_ptr->getJointsBounds()) {
      // each joint bound might have more than one variable
      for (auto&& bound : *joint_bound) {
        this->bounds_high.push_back(bound.max_position_);
        this->bounds_low.push_back(bound.min_position_);
      }
    }
    ////////////////////////////////////////
    ///// Get robot model for retrieving jacobian
    kinematic_state = std::make_shared<moveit::core::RobotState>(
        this->casted_space_ptr->getRobotModel());
    kinematic_state->setToDefaultValues();
    //
    joint_model_group = this->casted_space_ptr->getJointModelGroup();
    linkModels        = joint_model_group->getLinkModels();

    reference_point_position = Eigen::Vector3d(0.0, 0.0, 0.0);
  }

  ////////////////////////////////////////////////////////////
  moveit::core::RobotStatePtr kinematic_state;
  const moveit::core::JointModelGroup* joint_model_group;
  std::vector<const moveit::core::LinkModel*> linkModels;
  Eigen::Vector3d reference_point_position;
  //  Eigen::MatrixXd _jacobian;
  ////////////////////////////////////////////////////////////
  //  size_t m_thread_num;
  //  mutable ThreadPool m_thread_pool;
  ////////////////////////////////////////////////////////////

  moodycamel::ConcurrentQueue<std::array<double, NumDim>> m_difted_samples_Q;
  std::unique_ptr<std::thread> background_sampling_th;
  std::atomic<bool> m_finish_sampling;
  double m_radius_of_joint{0.05};
  ////////////////////////////////////////////////////////////

  void set_radius_of_joint(double r) { m_radius_of_joint = r; }

  bool sampleUniform_Diffeomorphic(ob::State* state) override {
#ifdef SXS_DIFF_SAMP_LOG_STATS
    auto& stats =
        sxs::g::get_stats(sxs::g::storage::get<std::thread::id>("thread_id"));

    //    std::ofstream stream(sxs::get_home_dir() + "/out_diff.csv",
    //                           std::ios::app | std::ios::out);
    //
    //    stream << stats.m_timer->elapsed() << "," <<
    //    stats.of<long>("used_drift")
    //        << ","
    //        << stats.of<long>("uniform") << "\n";

    //    stats.set_stats_output_file(sxs::Stats::get_default_fname() +
    //    "_haaa.csv"); stats.of<long>("samp_cnt") += 1; // this stat is already
    //    in the file

//    stats.of<long>("used_drift") += 0;
//    stats.of<long>("uniform") += 0;
//    stats.serialise_to_csv();
#endif
    std::array<double, NumDim> vals;
    bool ok = m_difted_samples_Q.try_dequeue(vals);
    if (!ok) {
#ifdef SXS_DIFF_SAMP_LOG_STATS
      stats.of<long>("uniform") += 1;
#endif
      this->sampleUniform_PurelyUniformStateSampler(state);
      return false;
    }
#ifdef SXS_DIFF_SAMP_LOG_STATS
    stats.of<long>("used_drift") += 1;
#endif
    /* assign rstate_values with pointer (the matrix NEEDS to be row-major) */
    // get raw pointer from the matrix
    double* rstate_values =
        state->as<ompl_interface::JointModelStateSpace::StateType>()->values;
    std::copy(std::begin(vals), std::end(vals), rstate_values);
    return true;
  }
  //  void sampleUniform_Diffeomorphic(ob::State* state) override {
  //    DEBUG_LOG("samp_diff ", cur_BATCH, " ", BATCH_A_cur_idx, " ",
  //              BATCH_B_cur_idx, " | ",
  //              BATCH_A_ready_idx.load(std::memory_order_relaxed), " ",
  //              BATCH_B_ready_idx.load(std::memory_order_relaxed));
  //    double* from;
  //    switch (cur_BATCH) {
  //      case 0:
  //        if (BATCH_A_cur_idx == this->rand_batch_sample_size - 1) {
  //          DEBUG_LOG("Resampling");
  //          // need to switch to a different batch
  //          cur_BATCH       = 1;
  //          from            = &BATCH_A(BATCH_A_cur_idx, 0);
  //          BATCH_A_cur_idx = 0;
  //          _start_sampling_batch(0);  // starts sampling in this batch
  //          break;
  //        } else if (BATCH_A_cur_idx >=
  //                   BATCH_A_ready_idx.load(std::memory_order_relaxed)) {
  //          //          // need to wait for the background thread to catch up
  //          //          // do a purely uniform sampling instead to avoid
  //          overhead this->sampleUniform_PurelyUniformStateSampler(state);
  //          return;
  //          //          std::cout << "waiting at " << cur_BATCH << std::endl;
  //          //          BATCH_A_lock.lock();
  //          //          BATCH_A_lock.unlock();
  //          //          std::cout << "waitdoneing at " << cur_BATCH <<
  //          std::endl;
  //        }
  //        // ok to go ahead
  //        from = &BATCH_A(BATCH_A_cur_idx++, 0);
  //        break;
  //      case 1:
  //        if (BATCH_B_cur_idx == this->rand_batch_sample_size - 1) {
  //          DEBUG_LOG("Resampling");
  //          // need to switch to a different batch
  //          cur_BATCH       = 0;
  //          from            = &BATCH_B(BATCH_B_cur_idx, 0);
  //          BATCH_B_cur_idx = 0;
  //          _start_sampling_batch(1);  // starts sampling in this batch
  //          break;
  //          //          sampleUniform_Diffeomorphic(state);  // call this
  //          function
  //          //          again return;
  //        } else if (BATCH_B_cur_idx >=
  //                   BATCH_B_ready_idx.load(std::memory_order_relaxed)) {
  //          //          // need to wait for the background thread to catch up
  //          //          // do a purely uniform sampling instead to avoid
  //          overhead this->sampleUniform_PurelyUniformStateSampler(state);
  //          return;
  //          //          std::cout << "waiting at " << cur_BATCH << std::endl;
  //          //          BATCH_B_lock.lock();
  //          //          BATCH_B_lock.unlock();
  //          //          std::cout << "done at " << cur_BATCH << std::endl;
  //        }
  //        // ok to go ahead
  //        from = &BATCH_B(BATCH_B_cur_idx++, 0);
  //        break;
  //    }
  //    auto& stats =
  //        sxs::g::get_stats(sxs::g::storage::get<std::thread::id>("thread_id"));
  //    stats.of<long>("used_drifted") += 1;
  //
  //    /* assign rstate_values with pointer (the matrix NEEDS to be row-major)
  //    */
  //    // get raw pointer from the matrix
  //    double* rstate_values =
  //        state->as<ompl_interface::JointModelStateSpace::StateType>()->values;
  //    std::copy(from, from + NumDim, rstate_values);
  //  }

  void finish_sampling() { m_finish_sampling.store(true); }

  void start_sampling() {
    if (!this->use_diff) return;
    background_sampling_th.reset(
        new std::thread([this] { _start_sampling(); }));
  }

  void _start_sampling() {
#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
    std::ofstream ofstream(sxs::get_home_dir() + "/out_tmp.csv",
                           std::ios::app | std::ios::out);
    auto si_ = sxs::g::storage::get<ob::SpaceInformationPtr>("si");

    //    auto motion = std::make_unique<og::RRTstarConnect::Motion>(si_);
    //    ob::State* allocated_state = si_->allocState();
    ompl_interface::JointModelStateSpace::StateType* allocated_state =
        si_->allocState()
            ->as<ompl_interface::JointModelStateSpace::StateType>();
    bool result_before;
    bool result_after;
//    sxs::g::storage::print_stored_info();
#endif
    // starts sampling in the background
    //    const size_t linkModel_start = 2;
    //    const size_t linkModel_end   = linkModels.size();

    /** the actual joint model of Jaco seems to starts from 1 to 7
     * i.e. joint 1 is the x-y plane rotational root joint.
     * However, since the jacobian looks something like
     * 0 0 0 0 0 0
     * 0 0 0 0 0 0
     * 0 0 0 0 0 0
     * 0 0 0 0 0 0
     * 0 0 0 0 0 0
     * 1.4e-16 0 0 0 0 0
     * it doesn't actually affect much of its previous joint (as it has no
     * previous joints.)
     * So we will only compute the link model of the rest of the joints
     */
    const size_t Actual_linkModel_start = 1;
    const size_t linkModel_start = Actual_linkModel_start + 1;  // skip first
    const size_t linkModel_end   = Actual_linkModel_start + 6;

    const size_t extra_bodypts_per_joint = 6;

    auto body_points_offset_mat = torch::tensor({{-m_radius_of_joint, 0., 0.},
                                                 {+m_radius_of_joint, 0., 0.},
                                                 {0., -m_radius_of_joint, 0.},
                                                 {0., +m_radius_of_joint, 0.},
                                                 {0., 0., -m_radius_of_joint},
                                                 {0., 0., +m_radius_of_joint}},
                                                torch::kFloat);
    // repeat this for all joints and qs
    //    body_points_offset_mat = body_points_offset_mat.repeat(
    //        {this->rand_batch_sample_size, (linkModel_end - linkModel_start),
    //        1,
    //         1});
    body_points_offset_mat = body_points_offset_mat.repeat(
        {this->rand_batch_sample_size * (linkModel_end - linkModel_start), 1});

    const MatrixXfRowMajor body_points_offset_matXXX = MatrixXfRowMajor::Map(
        body_points_offset_mat.data_ptr<float>(),
        body_points_offset_mat.size(0), body_points_offset_mat.size(1));

    // the following is the number of worldspace point per q
    // i.e., the number of joints X the number of body pts per joint
    const size_t size_per_q =
        (linkModel_end - linkModel_start) * extra_bodypts_per_joint;

    //    TorchOccMapManager::mutable_model_path() =
    //        "/home/soraxas/Downloads/diff_nn/occmap_cupboard_torchmodel.pt";
    //    TorchOccMapManager::mutable_model_path() =
    //        "/home/soraxas/Downloads/diff_nn/occmap_divider_torchmodel.pt";
    TorchOccMapManager::mutable_model_path() =
        "/home/soraxas/Downloads/diff_nn/occmap_real_torchmodel.pt";
    TorchOccMapManager::init();
    sxs::println("fixmeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");

//#pragma omp parallel num_threads(1)
#pragma omp parallel num_threads(9)
    {
      TorchOccMapManager::TorchOccMapPtr occ_map =
          TorchOccMapManager::get_occmap(omp_get_thread_num());

//      sxs::Timer timer;
//      timer.set_autoprint();
//#define TIMER_STAMP(str) timer.stamp(str);
//#undef TIMER_STAMP
#define TIMER_STAMP(str) ;

      TIMER_STAMP("prepare");
#ifdef SXS_DIFF_SAMP_LOG_STATS
      auto& stats =
          sxs::g::get_stats(sxs::g::storage::get<std::thread::id>("thread_id"));
#endif

      const auto robot_model = this->casted_space_ptr->getRobotModel();
      moveit::core::RobotState robot_state(robot_model);
      std::vector<double> joint_values(NumDim);

      thread_local Eigen::Rand::Vmt19937_64 rand_urng{std::random_device{}()};
      Eigen::Rand::UniformRealGen<double> rand_uni_gen;
      TIMER_STAMP("done prepare");
      while (!m_finish_sampling) {
        TIMER_STAMP("samp rand batch");
        SampledMatRowMajor batch =
            rand_uni_gen.template generate<SampledMatRowMajor>(
                this->rand_batch_sample_size, NumDim, rand_urng);
        // 0-1  =>  0-diff  =>  lower-upper
        batch = (batch.rowwise() * this->bounds_diff).rowwise() +
                this->bounds_lower;
        TIMER_STAMP("done samp rand batch");

#ifdef COLLECT_ANGLES_DISTRIBUTION

        enum CollectAnglesPriorType{Uniform, Mvn, GMM};
        CollectAnglesPriorType prior_type = Uniform;
        //        CollectAnglesPriorType prior_type = GMM;
        //        CollectAnglesPriorType prior_type = Mvn;

        switch (prior_type) {
          case Uniform:
            break;
          case Mvn: {
            Eigen::Matrix<double, 6, 1> __mu;
            Eigen::Matrix<double, 6, 1> __sigma;
            __mu << -1.3, 2.5, 6.1,  //
                0, 0, 0;
            __sigma << .4, .9, 3.1,  //
                3.5, 3.5, 3.5;

            Eigen::Rand::MvNormalGen<double> mvn_gen(
                __mu, Eigen::MatrixXd(__sigma.asDiagonal()));

            //        sxs::println(mvn_gen.generate(rand_urng,
            //        this->rand_batch_sample_size));
            //        sxs::println("--mvn_gen.template
            //        generate<SampledMatRowMajor>"
            //            "(rand_urng)");
            batch = mvn_gen.generate(rand_urng, this->rand_batch_sample_size)
                        .transpose();
          } break;
          case GMM: {
            SampledMatRowMajor mixture_1;
            SampledMatRowMajor mixture_2;
            SampledMatRowMajor mixture_3;
            {
              float prob = .5;
              Eigen::Matrix<double, 6, 1> __mu;
              Eigen::Matrix<double, 6, 1> __sigma;
              //////////////////////////////////////////
              __mu << -2.8, 2.5, 6.1,  //
                  0, 0, 0;
              __sigma << .1, .05, .4,  //
                  3.5, 3.5, 3.5;
              Eigen::Rand::MvNormalGen<double> mvn_gen(
                  __mu, Eigen::MatrixXd(__sigma.asDiagonal()));
              mixture_1 =
                  mvn_gen
                      .generate(rand_urng, this->rand_batch_sample_size * prob)
                      .transpose();
            }
            {
              float prob = .26;
              Eigen::Matrix<double, 6, 1> __mu;
              Eigen::Matrix<double, 6, 1> __sigma;
              //////////////////////////////////////////
              __mu << 1.1, 5.5, 3.5,  //
                  0, 0, 0;
              __sigma << .05, .5, .4,  //
                  3.5, 3.5, 3.5;
              Eigen::Rand::MvNormalGen<double> mvn_gen(
                  __mu, Eigen::MatrixXd(__sigma.asDiagonal()));
              mixture_2 =
                  mvn_gen
                      .generate(rand_urng, this->rand_batch_sample_size * prob)
                      .transpose();
            }
            {
              float prob = .24;
              Eigen::Matrix<double, 6, 1> __mu;
              Eigen::Matrix<double, 6, 1> __sigma;
              //////////////////////////////////////////
              __mu << 3.5, 1, 0.5,  //
                  0, 0, 0;
              __sigma << .1, .1, .6,  //
                  3.5, 3.5, 3.5;
              Eigen::Rand::MvNormalGen<double> mvn_gen(
                  __mu, Eigen::MatrixXd(__sigma.asDiagonal()));
              mixture_3 =
                  mvn_gen
                      .generate(rand_urng, this->rand_batch_sample_size * prob)
                      .transpose();
            }
            // concat all mixture
            SampledMatRowMajor mixture_all(
                mixture_1.rows() + mixture_2.rows() + mixture_3.rows(),
                mixture_1.cols());
            mixture_all << mixture_1, mixture_2, mixture_3;

            batch = mixture_all;
          } break;
        }
#endif

#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
        SampledMatRowMajor original_batch = batch;  // copy
#endif

        //        for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
        //          double* _from = &batch(i, 0);
        //          double* _to   = &vals[0];
        //          std::copy(_from, _from + NumDim, _to);
        //#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
        //          /* THIS IS IMPORTANT as we are re-using the same state.
        //          This clear
        //           * caches, or else it will always reports the same
        //           validity
        //           * regardless of chaning the values */
        //          allocated_state->clearKnownInformation();
        //          std::copy(_to, _to + NumDim, allocated_state->values);
        //          //        si_->printState(allocated_state);
        //          result_before = si_->isValid(allocated_state);
        //#endif
        //#ifdef SXS_LOG_TO_CSV
        //          std::stringstream ss;
        //          _log_to_stream(_to, linkModel_start, linkModels.size(),
        //          robot_state,
        //                         ss);
        //#endif
        //        }

//#define SXS_LOG_ALL_INTERMEDIATE_XS_QS
#ifdef SXS_LOG_ALL_INTERMEDIATE_XS_QS
        // the following is NOT thread safe
        std::ofstream myfile;
        myfile.open("/home/soraxas/example2.txt");
        myfile << "--------------\n";
#endif

        /* this is a list of Jacobian in the format of
         * j1 (of q1)
         * j2 (of q1)
         * ...
         * j6 (of q1)
         * j1 (of q2)
         * j2 (of q2)
         * ...
         * j6 (of qn)
         */
        // this list of jacobian will be reused within each drift
        //        std::vector<Eigen::MatrixXd> Jacobians;
        //        Jacobians.resize(this->rand_batch_sample_size *
        //                         (linkModel_end - linkModel_start));

        //  Matrix<double, -1, -1> m(6, 6 * batch_size);
        /** --------- ---------     ---------        ---------
         *  | q1-J1 | | q1-J2 | ... | q1-J6 | ...... | qn-J6 |
         *  --------- ---------     ---------        ---------
         */
        //        Eigen::MatrixXd Jacobians2(6, 6 *
        //        this->rand_batch_sample_size
        //        *
        //                                          (linkModel_end -
        //                                          linkModel_start));
        torch::Tensor Jacobians2 = torch::empty(
            {this->rand_batch_sample_size, (linkModel_end - linkModel_start),
             NumDim, WORLDSPACE_NUM_DIM},
            torch::kDouble);
        // ^^ we stores the transposed J, hence the swapped
        // [WORLDSPACE_NUM_DIM, NumDim] position (ie. 6,3 instead of 3,6)

        //        Eigen::MatrixXd Jacobians2 = Eigen::Matrix<double, -1,
        //        -1>::Random(6, 100); Jacobians2.resize

        //        sxs::println(Jacobians2);
        //        sxs::println("m.rows() ", Jacobians2.rows());
        //        sxs::println("m.cols() ", Jacobians2.cols());

        //  for (int i = 0; i < 2; ++i) {
        //    sxs::println(Jacobians2.middleCols(6 * i, 6));
        //    sxs::println("");
        //
        //  }

#ifdef COLLECT_ANGLES_DISTRIBUTION

        SampledMatRowMajor copied_batch            = batch;  // copy
        SampledMatRowMajor without_user_bias_batch = batch;  // copy
        SampledMatRowMajor with_user_bias          = batch;  // copy

        enum CollectAnglesAction{WithUserBiasFunction, WithoutUserBiasFunction,
                                 Last};

        for (int OperationTypeInt = WithUserBiasFunction;
             OperationTypeInt != Last; OperationTypeInt++) {
          CollectAnglesAction OperationType =
              static_cast<CollectAnglesAction>(OperationTypeInt);
          /////////////////////////////////////////////////

          batch = copied_batch;

          //          switch (OperationType) {
          //            case WithUserBiasFunction: {
          //              // first do a pass on attraction
          //
          //              SampledMatRowMajor target = batch;
          //              target.setZero();
          //              Eigen::Array<double, 1, 6> _target(6);
          //              _target << 0, //
          //                  (this->bounds_high[1] - this->bounds_low[1]) /
          //                  2,
          //                  // (this->bounds_high[2] - this->bounds_low[2])
          //                  / 2,  // 0, 0, 0;
          //              target.rowwise() = _target;
          //
          //              //            sxs::println(target);
          //              //
          //              //            sxs::println("-=----||------target");
          //              //            sxs::println(batch);
          //
          //              double _scaling_factor = 0.05;
          //              // the attraction function is x^2
          //              for (int _ = 0; _ < 3; ++_) {
          //                batch -= _scaling_factor * 2 * (batch - target);
          //              }
          //              //            sxs::println("-=----------target");
          //              //            sxs::println(batch);
          //            } break;
          //            case WithoutUserBiasFunction:
          //              break;
          //            default:
          //              throw std::runtime_error("unknown operation type");
          //          }

#endif

          for (size_t _drift = 0; _drift < this->num_drift; ++_drift) {
#ifdef SXS_LOG_ALL_INTERMEDIATE_XS_QS
            myfile << "drift " << _drift << "\n";
#endif

#ifdef COLLECT_ANGLES_DISTRIBUTION
            switch (OperationType) {
              case WithUserBiasFunction: {
                // first do a pass on attraction

                SampledMatRowMajor target = batch;
                target.setZero();
                Eigen::Array<double, 1, 6> _target(6);
                _target << 0,                                          //
                    (this->bounds_high[1] - this->bounds_low[1]) / 2,  //
                    (this->bounds_high[2] - this->bounds_low[2]) / 2,  //
                    0, 0, 0;
                target.rowwise() = _target;

                double _scaling_factor = 0.02;
                // the attraction function is x^2
                batch -= _scaling_factor * 2 * (batch - target);
                //            sxs::println("-=----------target");
                //            sxs::println(batch);
              } break;
              case WithoutUserBiasFunction:
                break;
              default:
                throw std::runtime_error("unknown operation type");
            }
#endif

            TIMER_STAMP("prepare to collect body points");
            // collects all body points
            MatrixXfRowMajor pts_to_get_grad(
                this->rand_batch_sample_size *
                    (linkModel_end - linkModel_start) * extra_bodypts_per_joint,
                3);
            //          std::vector<EigenWorldSpacePtVector>
            //          pts_to_get_grad;
            //          pts_to_get_grad.reserve(this->rand_batch_sample_size
            //          * size_per_q);

            for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
              double* _from = &batch(i, 0);
              TIMER_STAMP("set robot state");
              // set robot state to the desire values
              robot_state.setJointGroupPositions(joint_model_group, _from);
#ifdef SXS_LOG_ALL_INTERMEDIATE_XS_QS
              myfile << "joint idx " << i << ": " << joint_values << "\n";
#endif
              for (size_t j = 0; j < linkModel_end - linkModel_start; ++j) {
                size_t link_idx = j + linkModel_start;  // link index
                // we directly pass the jacobian stored in the list (as a
                // reference) to the function.

                TIMER_STAMP("get jacobian");
                Eigen::MatrixXd _jacobian;
                robot_state.getJacobian(joint_model_group, linkModels[link_idx],
                                        reference_point_position, _jacobian);

                // assign this local jacobian to the assembled big
                MatrixXdRowMajor::Map(Jacobians2
                                          .index({static_cast<int>(i),
                                                  static_cast<int>(j), "..."})
                                          .data_ptr<double>(),
                                      Jacobians2.size(-2),
                                      Jacobians2.size(-1)) =
                    _jacobian.topRows(3).transpose();

                TIMER_STAMP("get translation");
                Eigen::Vector3d pos =
                    robot_state.getGlobalLinkTransform(linkModels[link_idx])
                        .translation();
                TIMER_STAMP("push all body points");

                // insert the pos 6 times for body points
                //              pts_to_get_grad.insert(pts_to_get_grad.end(),
                //                                     extra_bodypts_per_joint,
                //                                     pos);

                int joint_offset = i * ((linkModel_end - linkModel_start) *
                                        extra_bodypts_per_joint) +
                                   j * extra_bodypts_per_joint;

                pts_to_get_grad
                    .middleRows(joint_offset, extra_bodypts_per_joint)
                    .rowwise() = pos.transpose().cast<float>();
                /*
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(0, 0) -= m_radius_of_joint;
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(0, 0) += m_radius_of_joint;
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(1, 0) -= m_radius_of_joint;
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(1, 0) += m_radius_of_joint;
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(2, 0) -= m_radius_of_joint;
                pts_to_get_grad.push_back(pos);
                pts_to_get_grad.back()(2, 0) += m_radius_of_joint;
                 */

                /*
                torch::Tensor pts_view = pts_to_get_grad2VVV.index(
                    {static_cast<int>(i), static_cast<int>(j), "..."});

                int _pt_idx          = 0;
                Eigen::Vector3f posf = pos.cast<float>();

                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 0}) -= m_radius_of_joint;
                ++_pt_idx;
                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 0}) += m_radius_of_joint;
                ++_pt_idx;
                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 1}) -= m_radius_of_joint;
                ++_pt_idx;
                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 1}) += m_radius_of_joint;
                ++_pt_idx;
                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 2}) -= m_radius_of_joint;
                ++_pt_idx;
                Eigen::Vector3f::Map(
                    pts_view.index({_pt_idx, torch::indexing::Slice()})
                        .data_ptr<float>()) = posf;
                pts_view.index({_pt_idx, 2}) += m_radius_of_joint;
                ++_pt_idx;
                 */

                TIMER_STAMP("done push all body points");
                /////////////////////////////////////////////
#ifdef SXS_LOG_ALL_INTERMEDIATE_XS_QS
                myfile << "x " << pos << "\n";
#endif
                /////////////////////////////////////////////
              }  // for each joint
            }    // for each sampled q

            pts_to_get_grad += body_points_offset_matXXX;
            //          sxs::println(body_points_offset_matXXX.rows());
            //          sxs::println(body_points_offset_matXXX.cols());

            torch::Tensor pts_to_get_grad2 = torch::from_blob(
                pts_to_get_grad.data(),
                {pts_to_get_grad.rows(), pts_to_get_grad.cols()});

            //          sxs::println(pts_to_get_grad2.sizes());

            //          pts_to_get_grad2 = pts_to_get_grad2.reshape(
            //              {this->rand_batch_sample_size, (linkModel_end -
            //              linkModel_start),
            //               6, 3});

            /*
            sxs::println("-------  pts -----------");
            sxs::println(pts_to_get_grad);
            sxs::println("-------  new -----------");
            sxs::println(pts_to_get_grad2.index(
                {-1, -1, torch::indexing::Slice(),
            torch::indexing::Slice()}));
            sxs::println(pts_to_get_grad2.sizes());
            sxs::println("-------  mm 2 -----------");
            sxs::println(pts_to_get_grad + body_points_offset_matXXX);
             */
            /*
            sxs::println(body_points_offset_mat.index(
                {-1, -1, torch::indexing::Slice(),
            torch::indexing::Slice()})); sxs::println("-------  new 2
            -----------"); torch::Tensor ttt = torch::from_blob(
                pts_to_get_grad.data(),
                {pts_to_get_grad.rows(), pts_to_get_grad.cols()});
            ttt = ttt.reshape({this->rand_batch_sample_size,
                               (linkModel_end - linkModel_start),
                               extra_bodypts_per_joint, 3});
            sxs::println(ttt.sizes());
            sxs::println(body_points_offset_mat.sizes());
            ttt += body_points_offset_mat;
            // offset the copied body points by joint radius
            sxs::println(ttt.index(
                {-1, -1, torch::indexing::Slice(),
            torch::indexing::Slice()}));
            */
            //          sxs::println("-------  done -----------");
            //          exit(1);

            //          if (pts_to_get_grad.size() !=
            //              this->rand_batch_sample_size * size_per_q) {
            //            throw std::runtime_error(
            //                "The size of points to get gradient does"
            //                " not match what is expected");
            //          }

            TIMER_STAMP("pass through nn");
            // ========== drift =============
            // points to tensor
            //          auto tensor2 =
            //              TorchOccMap::vector_of_eigen_vec_to_libtorch(pts_to_get_grad)
            //                  .to(occ_map->device);
            //          auto old_grad2_as_tensor = occ_map->grad2(tensor2) *
            //          1e-3;

            //          auto grad2_as_tensor =
            //              occ_map->grad2(pts_to_get_grad2.reshape({-1,
            //              WORLDSPACE_NUM_DIM})
            //                                 .to(occ_map->device)) *
            //              1e-3;
            auto grad2_as_tensor =
                occ_map->grad2(pts_to_get_grad2.to(occ_map->device)) * 1e-3;
            TIMER_STAMP("done pass through nn");
            /////////////////
            /*
            // ========== Consolidate =============
            // --- for each q
            for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
              double* _from   = &batch(i, 0);
              size_t q_offset = size_per_q * i;
              // --- for each joint
              for (size_t j = 0; j < linkModel_end - linkModel_start; ++j) {
                size_t l_idx = j + linkModel_start;  // link index

                TIMER_STAMP("reduce to consolidate grad across body
  points"); size_t joint_offset = q_offset + j * extra_bodypts_per_joint;
                torch::Tensor consolidated_grad =
                    old_grad2_as_tensor
                        .index({torch::indexing::Slice(
                                    joint_offset,
                                    joint_offset + extra_bodypts_per_joint),
                                torch::indexing::Slice()})
                        .mean(0);
                TIMER_STAMP("done consolidate grad");

                //              sxs::println("+++++ ori ++++++++=");
                //              sxs::println(consolidated_grad);
                //                            sxs::println("+++++ new
  ++++++++=");
                //              sxs::println(ohno.index({int(i), int(j),
  "..."}));
                //                            sxs::println("+++++ done
                //                            ++++++++=");

                ////////////////
                //              torch::Tensor consolidated_grad =
                //              old_grad2_as_tensor.index(
                //                  {torch::indexing::Slice(
                //                       joint_offset, joint_offset +
                //                       extra_bodypts_per_joint),
                //                   torch::indexing::Slice()});
                //
                //              // argmax of squared norm
                //              consolidated_grad = consolidated_grad.index(
                // {consolidated_grad.pow(2).sum(1).argmax(),
                //                   torch::indexing::Slice()});
                ////////////////

                //              sxs::println("q_offset ", q_offset);
                //              sxs::println("joint_offset ", joint_offset);
                //                            sxs::println("============");
                // sxs::println(old_grad2_as_tensor);
                //                            sxs::println("-------------");
                //                            sxs::println(joint_offset);
                // sxs::println(extra_bodypts_per_joint);
                //                            sxs::println("-------------");
                // sxs::println(old_grad2_as_tensor
                //                      .index({torch::indexing::Slice(
                //                                  joint_offset,
                //                                  joint_offset +
                // extra_bodypts_per_joint),
                //                              torch::indexing::Slice()}));
                //                            sxs::println("-------------");
                // sxs::println("consolidated_grad
  ",
                //                            consolidated_grad);
                //
                //                            exit(1);

                // ========== dot product =============
                // dot it with jacobian to get cspace gradients
                // convert grad in worldspace to cspace

                //              sxs::println(i * (linkModel_end -
  linkModel_start)
                //              + j); sxs::println(
                //                  Jacobians[i * (linkModel_end -
                //                  linkModel_start) + j]);
                //              sxs::println("");

                //              sxs::println("+++++ ori ++++++++=");
                //              sxs::println(
                //                  Jacobians[i * (linkModel_end -
                //                  linkModel_start) + j]);
                //              sxs::println("+++++ new ++++++++=");
                //              sxs::println(Jacobians2.index({int(i),
  int(j),
                //              "..."})); sxs::println("+++++ done
  ++++++++=");

                TIMER_STAMP("convert grad to cspace");
                Eigen::VectorXd grad_in_cspace =
                    Jacobians[i * (linkModel_end - linkModel_start) + j]
                        .block(0, 0, WORLDSPACE_NUM_DIM, NumDim)
                        .transpose() *
                    EigenWorldSpacePtVector::Map(
                        consolidated_grad
                            .to(torch::kCPU, torch::kDouble)
                            // .contiguous()  // it should already
  contiguous .data_ptr<double>(),
                        EigenWorldSpacePtVector::RowsAtCompileTime);

                TIMER_STAMP("epsilon step");
                Eigen::Array<double, 1, NumDim>::Map(_from, NumDim) +=
                    this->epsilon * grad_in_cspace.array();
                ///////////////////////////////////////
                _clamp_pt_jaco_arm(_from);
                ///////////////////////////////////////
                TIMER_STAMP("done epsilon step");

  #ifdef SXS_LOG_TO_CSV
  //              _log_to_stream(_from, linkModel_start, linkModels.size(),
  //                             robot_state, ss);
  #endif
              }  // for each joint
            }    // for each q

             */

            torch::Tensor reshaped_tensor =
                grad2_as_tensor
                    .reshape({this->rand_batch_sample_size,
                              (linkModel_end - linkModel_start),
                              extra_bodypts_per_joint, WORLDSPACE_NUM_DIM})
                    .mean(2);

            // back to original shape, sum across joints
            auto summed_grad =
                Jacobians2.to(occ_map->device)
                    .matmul(reshaped_tensor.unsqueeze(-1).to(torch::kDouble))
                    .sum(1)
                    .squeeze()
                    .to(torch::kCPU)
                    .contiguous();

            // take epsilon step
            //            batch.array() +=
            batch.array() -=
                this->epsilon *
                ArrayXdRowMajor::Map(summed_grad.data_ptr<double>(),
                                     summed_grad.size(0), summed_grad.size(1));

            /////////////////////////////////////////
            // clamp to boundary (the following two joints are specific to
            // jaco)
            batch.col(1) = batch.col(1)
                               .cwiseMin(this->bounds_high[1])
                               .cwiseMax(this->bounds_low[1]);
            batch.col(2) = batch.col(2)
                               .cwiseMin(this->bounds_high[2])
                               .cwiseMax(this->bounds_low[2]);
            /////////////////////////////////////////

            //          sxs::println("============batch -
            //          copied_batch============"); sxs::println((batch -
            //          copied_batch)); sxs::println((batch -
            //          copied_batch).square().colwise().sum());
            //          sxs::println(copied_batch.col(1).maxCoeff());
            //          sxs::println(copied_batch.col(1).minCoeff());
            //          sxs::println(copied_batch.col(2).maxCoeff());
            //          sxs::println(copied_batch.col(2).minCoeff());
            //          sxs::println("============-00000000000============");

          }  // for each drift

#ifdef COLLECT_ANGLES_DISTRIBUTION
          switch (OperationType) {
            case WithUserBiasFunction:
              with_user_bias = batch;
              break;
            case WithoutUserBiasFunction:
              without_user_bias_batch = batch;
              break;
            default:
              throw std::runtime_error("unknown operation type");
          }
        }
        /////////////////////////////////////////////////////
        std::ofstream stream("/home/soraxas/drifted_feasibility.csv",
                             std::ios::app | std::ios::out);

        auto log_q = [&stream, &allocated_state, &si_](SampledMatRowMajor& m,
                                                       int i) {
          allocated_state->clearKnownInformation();
          std::copy(&m(i, 0), &m(i, 0) + NumDim, allocated_state->values);
          bool okk = int(si_->isValid(allocated_state));
          stream << okk << "," << m(i, 0) << "," << m(i, 1) << "," << m(i, 2)
                 << "," << m(i, 3) << "," << m(i, 4) << "," << m(i, 5);
        };

        for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
          stream << "UNDRIFTED:,";
          log_q(original_batch, i);
          stream << ",DRIFTED,";
          log_q(without_user_bias_batch, i);
          stream << ",BIAS:,";
          log_q(with_user_bias, i);
          stream << "\n";
        }
#endif

        /////////////////////////////////////////////////////

        //        batch = copied_batch;

        TIMER_STAMP("done one drift");

#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
        // compare the before and after
        /////////////////////////
        for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
          // before
          allocated_state->clearKnownInformation();
          std::copy(&original_batch(i, 0), &original_batch(i, 0) + NumDim,
                    allocated_state->values);
          bool _result_before = si_->isValid(allocated_state);

          allocated_state->clearKnownInformation();
          std::copy(&batch(i, 0), &batch(i, 0) + NumDim,
                    allocated_state->values);
          bool _result_after = si_->isValid(allocated_state);

          //          if (_result_before && !_result_after) {
          //            std::cout << "====original_batch===" << std::endl;
          //            for (int j = 0; j < 6; ++j)
          //              sxs::print(original_batch(i, j) * 56, ", ");
          //            sxs::println("");
          //            std::cout << "====batch===" << std::endl;
          //            for (int j = 0; j < 6; ++j) sxs::print(batch(i, j) *
          //            56,
          //            ", "); sxs::println("");
          //          }

#ifdef SXS_LOG_ALL_INTERMEDIATE_XS_QS
          myfile << "validity before for q " << i << _result_before << "\n";
          myfile << "validity after for q " << i << _result_after << "\n";
#endif
#ifdef SXS_DIFF_SAMP_LOG_STATS
          stats.of<int>("drifted_cnt") += 1;
          stats.of<int>("total_0->1") +=
              static_cast<int>(!_result_before && _result_after);
          stats.of<int>("total_1->0") +=
              static_cast<int>(_result_before && !_result_after);
          stats.of<double>("total_pct_0->1") =
              static_cast<double>(stats.of<int>("total_0->1")) * 100. /
              stats.of<int>("drifted_cnt");
          stats.of<double>("total_pct_1->0") =
              static_cast<double>(stats.of<int>("total_1->0")) * 100. /
              stats.of<int>("drifted_cnt");

          stats.of<double>("drifted_pct") =
              static_cast<double>(stats.of<int>("drifted_cnt")) * 100. /
              stats.of<long>("samp_cnt");
#endif
        }
        /////////////////////////

#endif
        TIMER_STAMP("bulk enqueue");
        // TODO operate in-place for the queue
        // we are able to reinterpert the pointer as array pointer of fix size
        // because the eigen matrix is continguous along row
        m_difted_samples_Q.enqueue_bulk(
            reinterpret_cast<std::array<double, NumDim>*>(&batch(0, 0)),
            this->rand_batch_sample_size);
        TIMER_STAMP("done bulk enqueue");
#ifdef SXS_LOG_TO_CSV
//        /* ================================================== */
//        ofstream << static_cast<int>(result_before) << " "
//                 << static_cast<int>(result_after) << std::endl;
//        ofstream << ss.str();
//        ofstream << "done" << std::endl;
//        /* ================================================== */
#endif
      }
      DEBUG_LOG("Done one batch");
    }
    //    }

    //#pragma omp parallel num_threads(1)
    //    {
    //      auto& stats =
    //          sxs::g::get_stats(sxs::g::storage::get<std::thread::id>("thread_id"));
    //
    //      const auto robot_model = this->casted_space_ptr->getRobotModel();
    //      moveit::core::RobotState robot_state(robot_model);
    //      std::vector<double> joint_values(NumDim);
    //
    //      thread_local Eigen::Rand::Vmt19937_64
    //      rand_urng{std::random_device{}()};
    //      Eigen::Rand::UniformRealGen<double> rand_uni_gen;
    //      sxs::Timer timer;
    //      timer.set_autoprint();
    //      while (!m_finish_sampling) {
    //        SampledMatRowMajor batch =
    //            rand_uni_gen.template generate<SampledMatRowMajor>(
    //                this->rand_batch_sample_size, NumDim, rand_urng);
    //        // 0-1  =>  0-diff  =>  lower-upper
    //        batch = (batch.rowwise() * this->bounds_diff).rowwise() +
    //                this->bounds_lower;
    //
    //        std::array<double, NumDim> vals;
    //
    //        for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
    //          double* _from = &batch(i, 0);
    //          double* _to   = &vals[0];
    //          std::copy(_from, _from + NumDim, _to);
    //#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
    //          /* THIS IS IMPORTANT as we are re-using the same state. This
    //          clear
    //           * caches, or else it will always reports the same validity
    //           * regardless of chaning the values */
    //          allocated_state->clearKnownInformation();
    //          std::copy(_to, _to + NumDim, allocated_state->values);
    //          //        si_->printState(allocated_state);
    //          result_before = si_->isValid(allocated_state);
    //#endif
    //#ifdef SXS_LOG_TO_CSV
    //          std::stringstream ss;
    //          _log_to_stream(_to, linkModel_start, linkModels.size(),
    //                         robot_state, ss);
    //#endif
    //          for (size_t _drift = 0; _drift < this->num_drift; ++_drift) {
    //            ///////////////////////////////////////
    //            std::copy(_to, _to + NumDim, joint_values.begin());
    //            auto _out =
    //                _joint_state_to_grad(linkModel_start, linkModel_end,
    //                                     robot_state, joint_values, &timer);
    //            // get gradient at those points
    //            std::vector<EigenWorldSpacePtVector> gradient = _out;
    //            ////////////////////////////////////////////////////////////
    //            //    sxs::g::stats.of<int>("samp") += 1;
    //            //    auto& state =
    //            //
    //            sxs::g::get_deref<ompl_interface::JointModelStateSpace::StateType>(
    //            //            "state");
    //            ////////////////////////////////////////////////////////////
    //
    //            // dot it with jacobian to get cspace gradients
    //            Eigen::MatrixXd _jacobian;
    //            for (size_t l_idx = linkModel_start; l_idx <
    //            linkModel_end;
    //                 ++l_idx) {
    //              robot_state.getJacobian(joint_model_group,
    //              linkModels[l_idx],
    //                                      reference_point_position,
    //                                      _jacobian);
    //              // convert grad in worldspace to cspace
    //              auto grad_in_cspace =
    //                  _jacobian.block(0, 0, WORLDSPACE_NUM_DIM, NumDim)
    //                      .transpose() *
    //                  EigenWorldSpacePtVector::Map(
    //                      gradient[l_idx - linkModel_start].data(),
    //                      EigenWorldSpacePtVector::RowsAtCompileTime);
    //              Eigen::Array<double, 1, NumDim>::Map(_to, NumDim) -=
    //                  this->epsilon * grad_in_cspace.array();
    //            }
    //            ///////////////////////////////////////
    //            _clamp_pt_jaco_arm(_to);
    //            ///////////////////////////////////////
    //#ifdef SXS_LOG_TO_CSV
    //            _log_to_stream(_to, linkModel_start, linkModels.size(),
    //                           robot_state, ss);
    //#endif
    //          }
    //          // TODO operate in-place for the queue
    //          m_difted_samples_Q.enqueue(vals);
    //#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
    //          /////////////////////////
    //          allocated_state->clearKnownInformation();
    //          std::copy(_to, _to + NumDim, allocated_state->values);
    //          result_after = si_->isValid(allocated_state);
    //
    //          stats.of<int>("drifted_cnt") += 1;
    //          stats.of<int>("total_0->1") +=
    //              static_cast<int>(!result_before && result_after);
    //          stats.of<int>("total_1->0") +=
    //              static_cast<int>(result_before && !result_after);
    //          stats.of<double>("total_pct_0->1") =
    //              static_cast<double>(stats.of<int>("total_0->1")) * 100. /
    //              stats.of<int>("drifted_cnt");
    //          stats.of<double>("total_pct_1->0") =
    //              static_cast<double>(stats.of<int>("total_1->0")) * 100. /
    //              stats.of<int>("drifted_cnt");
    //
    //          stats.of<double>("drifted_pct") =
    //              static_cast<double>(stats.of<int>("drifted_cnt")) * 100. /
    //              stats.of<long>("samp_cnt");
    //          /////////////////////////
    //#endif
    //#ifdef SXS_LOG_TO_CSV
    //          /* ================================================== */
    //          ofstream << static_cast<int>(result_before) << " "
    //                   << static_cast<int>(result_after) << std::endl;
    //          ofstream << ss.str();
    //          ofstream << "done" << std::endl;
    //          /* ================================================== */
    //#endif
    //        }
    //        std::cout << "done 1 batch" << std::endl;
    //        DEBUG_LOG("Done one batch");
    //      }
    //    }

#ifdef SXS_LOG_TO_CSV_WITH_DRIFT_VALIDITY_RESULT
    si_->freeState(allocated_state);
#endif
    std::cout << "background sampling stopped" << std::endl;
  }

  inline void _clamp_pt_jaco_arm(double* values_ptr) const {
    // for jaco arm
    for (size_t i = 0; i < NumDim; ++i) {
      switch (i) {
        case 0:
        case 3:
        case 4:
        case 5:
          break;
          // 0 to 2pi (wrap around)
          /*
          if (values_ptr[i] < this->bounds_low[i])
            values_ptr[i] += 2 * M_PI;
          else if (values_ptr[i] > this->bounds_high[i])
            values_ptr[i] -= 2 * M_PI;
          break;
          */
        case 1:
        case 2:
          // clamp to lower boundary
          const double t = values_ptr[i] < this->bounds_low[i]
                               ? this->bounds_low[i]
                               : values_ptr[i];
          // clamp to upper boundary
          values_ptr[i] = t > this->bounds_high[i] ? this->bounds_high[i] : t;

          // fixed bounds
          //          // clamp to boundary
          //          if (values_ptr[i] < this->bounds_low[i]) values_ptr[i] =
          //              this->bounds_low[i];
          //          else if (values_ptr[i] > this->bounds_high[i])
          //          values_ptr[i] =
          //              this->bounds_high[i];
          //          // fixed bounds
      }
    }
  }

#define SXS_DIFF_USE_EXTRA_BODY_POINTS

  inline auto _joint_state_to_grad(const size_t linkModel_start,
                                   const size_t linkModel_end,
                                   moveit::core::RobotState& robot_state,
                                   const std::vector<double>& joint_state,
                                   sxs::Timer* timer = nullptr) const {
    // set robot state to the desire values
    robot_state.setJointGroupPositions(joint_model_group, joint_state);
    // collects all body points
    std::vector<EigenWorldSpacePtVector> pts_to_get_grad;
#ifdef SXS_DIFF_USE_EXTRA_BODY_POINTS
    const uint extra_body_pts_per_joint = WORLDSPACE_NUM_DIM * 2;
    pts_to_get_grad.reserve((linkModel_end - linkModel_start) *
                            extra_body_pts_per_joint);
#else
    pts_to_get_grad.reserve((linkModels.size() - linkModel_start));
#endif
    for (size_t link_idx = linkModel_start; link_idx < linkModel_end;
         ++link_idx) {
      pts_to_get_grad.push_back(
          robot_state.getGlobalLinkTransform(linkModels[link_idx])
              .translation());
    }
#ifdef SXS_DIFF_USE_EXTRA_BODY_POINTS
    // this container keep track of which index of joints does the
    // `pts_to_get_grad` belongs to

    // a vector of vector of long, where the outer vector's index refers to
    // joints, and the inner vector refers to the indexes
    // within `pts_to_get_grad` that belongs to that index
    std::vector<std::vector<long>> collection_of_joint_idx;
    collection_of_joint_idx.resize(linkModel_end - linkModel_start, {});

    std::vector<size_t> pts_to_get_grad__idx_of_joint;
    pts_to_get_grad__idx_of_joint.reserve((linkModel_end - linkModel_start) *
                                          extra_body_pts_per_joint);
    // index of the center of the joints
    for (uint i = 0; i < (linkModel_end - linkModel_start); ++i) {
      pts_to_get_grad__idx_of_joint.emplace_back(i);
      collection_of_joint_idx[i].push_back(i);
    }
    for (size_t body_pt_idx = 0; body_pt_idx < linkModel_end - linkModel_start;
         ++body_pt_idx) {
      // copy
      const EigenWorldSpacePtVector& joint_pt = pts_to_get_grad[body_pt_idx];
      for (int xyz = 0; xyz < 3; ++xyz) {
        {
          collection_of_joint_idx[body_pt_idx].push_back(
              pts_to_get_grad.size());

          pts_to_get_grad.push_back(joint_pt);  // copied
          pts_to_get_grad.back()(xyz, 0) -= m_radius_of_joint;
          // index of the surrounding body points of the given joint
          pts_to_get_grad__idx_of_joint.emplace_back(body_pt_idx);
        }
        {
          collection_of_joint_idx[body_pt_idx].push_back(
              pts_to_get_grad.size());

          pts_to_get_grad.push_back(joint_pt);  // copied
          pts_to_get_grad.back()(xyz, 0) += m_radius_of_joint;
          // index of the surrounding body points of the given joint
          pts_to_get_grad__idx_of_joint.emplace_back(body_pt_idx);
        }
      }
    }
    assert(pts_to_get_grad.size() == pts_to_get_grad__idx_of_joint.size());
#endif
#ifdef SXS_USE_TIMER
    std::shared_ptr<sxs::Timer> timer_ptr;
    if (!timer) {
      timer_ptr = std::make_shared<sxs::Timer>();
      timer     = timer_ptr.get();
    }
#endif
    sxs::Timer timerrr;
    //    timerrr.stamp("hilbertmap");
    //    auto res =
    //        this->m_hilbert_map->m_map->occupancy_and_gradient(pts_to_get_grad
    //#ifdef SXS_USE_TIMER
    //                                                           ,
    //                                                           timer
    //#endif
    //        );
    //    timerrr.stamp("hilbertmap done");

    /*
    std::cout << "========= pts as tensor ===========" << std::endl;
    auto tensor =
    TorchOccMap::vector_of_eigen_vec_to_libtorch(pts_to_get_grad); std::cout
    << tensor << std::endl; std::cout << "========= hilbertmap grad
    ===========" << std::endl; for (auto&& vec : res.second) std::cout <<
    vec.transpose() << std::endl; std::cout << "========= occ map grad
    ===========" << std::endl;

    timerrr.stamp("occmap");
    auto tensor2 =
        TorchOccMap::vector_of_eigen_vec_to_libtorch(pts_to_get_grad);
    timerrr.stamp("to device");
    tensor2 = tensor2.to(this->m_occ_map->device);
    timerrr.stamp("convert done");
    auto old_grad2_as_tensor =
        this->m_occ_map->grad2(tensor2.repeat({10, 1})) * 1e-3;
    timerrr.stamp("occmap done");

    std::cout << old_grad2_as_tensor << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << pts_to_get_grad__idx_of_joint << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << collection_of_joint_idx << std::endl;
     */

    //    auto tt = torch::from_blob(vvv.data(), vvv.size(),
    //                               torch::TensorOptions().dtype(torch::kLong));

#ifdef SXS_DIFF_USE_EXTRA_BODY_POINTS
    // collect results from the extra body points
    //    std::vector<DefaultHM::DataType> consolidated_grad(linkModel_end -
    //                                                       linkModel_start);

    std::vector<EigenWorldSpacePtVector> all_consolidated_grad;
    all_consolidated_grad.reserve(linkModel_end - linkModel_start);

    for (uint link = 0; link < linkModel_end - linkModel_start; ++link) {
      std::vector<EigenWorldSpacePtVector> consolidated_grad;
      for (uint pt_idx = 0; pt_idx < pts_to_get_grad__idx_of_joint.size();
           ++pt_idx) {
        //        if (pts_to_get_grad__idx_of_joint[pt_idx] == link)
        //          consolidated_grad.push_back(res.second[pt_idx]);
      }

      // the following will copy the matrix :L
      all_consolidated_grad.push_back(
          *std::max_element(consolidated_grad.begin(), consolidated_grad.end(),
                            [](auto const& lhs, auto const& rhs) {
                              return lhs.squaredNorm() < rhs.squaredNorm();
                            }));
      //      all_consolidated_grad.push_back(sxs::vec::sum(consolidated_grad));
    }

    std::vector<EigenWorldSpacePtVector> all_consolidated_grad2;
    all_consolidated_grad.reserve(linkModel_end - linkModel_start);

    for (uint i = 0; i < all_consolidated_grad.size(); ++i) {
      auto tensor_index = torch::from_blob(
          collection_of_joint_idx[i].data(), collection_of_joint_idx[i].size(),
          torch::TensorOptions().dtype(torch::kLong));
      /*
      std::cout << "VVVVVVVVVVVVVvvvv" << std::endl;
      std::cout << old_grad2_as_tensor.index(
                       {tensor_index, torch::indexing::Slice()})
                << std::endl;
      std::cout << old_grad2_as_tensor
                       .index({tensor_index, torch::indexing::Slice()})
                       .mean(0, true)
                << std::endl;
      std::cout << "VVVVVVVVVVVVVvvvv" << std::endl;
       */

      torch::Tensor consolidated_grad = torch::rand({1, 2});
      //
      //      torch::Tensor consolidated_grad =
      //          old_grad2_as_tensor.index({tensor_index,
      //          torch::indexing::Slice()})
      //              .mean(0);

      Eigen::Map<EigenWorldSpacePtVector> _as_eigen(
          consolidated_grad.to(torch::kDouble).data_ptr<double>(),
          consolidated_grad.size(0));

      //      std::cout << _as_eigen << std::endl;
      //      all_consolidated_grad2.emplace_back(_as_eigen);

      all_consolidated_grad2.emplace_back(Eigen::Map<EigenWorldSpacePtVector>(
          consolidated_grad.to(torch::kCPU, torch::kDouble)
              .contiguous()
              .data_ptr<double>(),
          consolidated_grad.size(0)));
    }
    return all_consolidated_grad2;

    std::cout << all_consolidated_grad2 << std::endl;

    exit(1);

    //    return decltype(res){res.first, all_consolidated_grad};
    return all_consolidated_grad;
#else
    return res;
#endif
  }

  inline void _drift_for_one_pt_array(
      double* vals, const size_t linkModel_start, const size_t linkModel_end,
      moveit::core::RobotState& _tmp_robot_state,
      std::vector<double>& _tmp_joint_state) const {
    DEBUG_LOG("Drift idx ", _samp_idx);
    // assign sampled joints to stdvector
    std::copy(vals, vals + NumDim, _tmp_joint_state.begin());
    auto _out = _joint_state_to_grad(linkModel_start, linkModel_end,
                                     _tmp_robot_state, _tmp_joint_state);
    // get gradient at those points
    std::vector<EigenWorldSpacePtVector> gradient = _out;
    ////////////////////////////////////////////////////////////
    //    sxs::g::stats.of<int>("samp") += 1;
    //    auto& state =
    //        sxs::g::get_deref<ompl_interface::JointModelStateSpace::StateType>(
    //            "state");
    ////////////////////////////////////////////////////////////

    // dot it with jacobian to get cspace gradients
    Eigen::MatrixXd _jacobian;
    for (size_t l_idx = linkModel_start; l_idx < linkModel_end; ++l_idx) {
      _tmp_robot_state.getJacobian(joint_model_group, linkModels[l_idx],
                                   reference_point_position, _jacobian);
      // convert grad in worldspace to cspace
      auto grad_in_cspace =
          _jacobian.block(0, 0, WORLDSPACE_NUM_DIM, NumDim).transpose() *
          EigenWorldSpacePtVector::Map(
              gradient[l_idx - linkModel_start].data(),
              EigenWorldSpacePtVector::RowsAtCompileTime);
      Eigen::Array<double, 1, NumDim>::Map(vals, NumDim) +=
          this->epsilon * grad_in_cspace.array();
    }
    _clamp_pt_jaco_arm(vals);
  }

  inline void _drift_for_one_pt(const size_t _samp_idx,
                                SampledMatRowMajor& _samp_batch,
                                const size_t linkModel_start,
                                const size_t linkModel_end,
                                moveit::core::RobotState& _tmp_robot_state,
                                std::vector<double>& _tmp_joint_state) const {
    DEBUG_LOG("Drift idx ", _samp_idx);
    // assign sampled joints to stdvector
    Eigen::Array<double, 1, NumDim>::Map(_tmp_joint_state.data(), NumDim) =
        _samp_batch.block(_samp_idx, 0, 1, NumDim);
    auto _out = _joint_state_to_grad(linkModel_start, linkModel_end,
                                     _tmp_robot_state, _tmp_joint_state);
    // get gradient at those points
    std::vector<EigenWorldSpacePtVector> gradient = _out;
    ////////////////////////////////////////////////////////////
    //    sxs::g::stats.of<int>("samp") += 1;
    //    auto& state =
    //        sxs::g::get_deref<ompl_interface::JointModelStateSpace::StateType>(
    //            "state");
    ////////////////////////////////////////////////////////////

    // dot it with jacobian to get cspace gradients
    Eigen::MatrixXd _jacobian;
    for (size_t l_idx = linkModel_start; l_idx < linkModel_end; ++l_idx) {
      _tmp_robot_state.getJacobian(joint_model_group, linkModels[l_idx],
                                   reference_point_position, _jacobian);
      // convert grad in worldspace to cspace
      auto grad_in_cspace =
          _jacobian.block(0, 0, WORLDSPACE_NUM_DIM, NumDim).transpose() *
          EigenWorldSpacePtVector::Map(
              gradient[l_idx - linkModel_start].data(),
              EigenWorldSpacePtVector::RowsAtCompileTime);
      _samp_batch.block(_samp_idx, 0, 1, NumDim) -=
          this->epsilon * grad_in_cspace.array();
    }
    _clamp_pt_jaco_arm(&_samp_batch(_samp_idx, 0));
  }

  void _log_to_stream(double* from, const size_t linkModel_start,
                      const size_t linkModel_end,
                      moveit::core::RobotState& _tmp_robot_state,
                      std::ostream& stream) {
    std::vector<double> _tmp_joint_state(NumDim);
    std::copy(from, from + NumDim, _tmp_joint_state.data());
    auto _out = _joint_state_to_grad(linkModel_start, linkModel_end,
                                     _tmp_robot_state, _tmp_joint_state);

    //    stream << "occ: " << _out.first << std::endl;
    //    stream << "    sum: " << sxs::vec::sum(_out.first) << std::endl;
    //    stream << "    max: " << sxs::vec::max(_out.first) << std::endl;
    stream << "gra: " << eigen_as_vec(sxs::vec::sum(_out)) << std::endl;
  }

  void _log_to_stream(const size_t _samp_idx, SampledMatRowMajor& _samp_batch,
                      const size_t linkModel_start, const size_t linkModel_end,
                      moveit::core::RobotState& _tmp_robot_state,
                      std::ostream& stream) {
    _log_to_stream(&_samp_batch(_samp_idx, 0), linkModel_start, linkModel_end,
                   _tmp_robot_state, stream);
  }

  void drift_states() override { throw std::runtime_error("not in-use"); }

  void drift_states_sequential() {
    DEBUG_LOG("==== moveit drift ====");

    size_t linkModel_start = 2;
    std::vector<double> joint_values(NumDim);

    sxs::Timer timer;
    timer.set_autoprint();

    //    cppm::range(this->rand_sampled_batch);
    for (size_t i = 0; i < this->rand_batch_sample_size; ++i) {
      _drift_for_one_pt(i, this->rand_sampled_batch, linkModel_start,
                        linkModels.size(), *kinematic_state, joint_values);
    }
  }
};
#endif
