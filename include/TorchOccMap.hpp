#pragma once

#include <torch/script.h> // One-stop header.

//#include <torch/torch.h>
#include <ATen/Parallel.h>
#include <torch/cuda.h>

//#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <string>

#include <iostream>
#include <memory>

#include <Eigen/Dense>

template <typename V>
using MatrixXrm =
    typename Eigen::Matrix<V, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class TorchOccMap {
public:
  using EigenWorldspace = Eigen::Matrix<double, 3, 1>;

  /**
   *
   * @param model_path - the path to the serialised model
   */
  explicit TorchOccMap(const std::string &model_path)
      : device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
        // device_idx(0),
        high_stream_priority(true),
        cuda_stream(c10::cuda::getStreamFromPool(high_stream_priority, 0)) {
    // for now we will enforce the use of cuda
    if (!torch::cuda::is_available())
      throw std::runtime_error("CUDA is not available!");

    ///////////////////////////////////////////

    m_model = torch::jit::load(model_path);
    m_model.to(device, torch::kFloat);
    m_model.eval();
    // set all parameters to be not require grad
    for (auto &&param : m_model.parameters(true)) {
      param.requires_grad_(false);
    }
  }

  /**
   *
   * @return the internal torchscript model
   */
  torch::jit::script::Module &model() { return m_model; }

  /**
   * Using analytical gradient
   *
   * @param input - [N x 3] coordinate
   * @return [N x 3] gradient wrt coordinate
   */
  torch::Tensor grad1(torch::Tensor &input) {
    // guard active for this scope
    c10::cuda::CUDAStreamGuard guard(cuda_stream);

    return m_model.run_method("grad", input).toTensor();
  }

  /**
   *
   * @param input - [N x 3] coordinate
   * @return [N x 3] gradient wrt coordinate
   */
  torch::Tensor grad2(const torch::Tensor &input) {
    // guard active for this scope
    c10::cuda::CUDAStreamGuard guard(cuda_stream);

    input.requires_grad_(true);
    auto out = m_model.run_method("forward_without_sigmoid", input).toTensor();
    // std::cout << out.grad_fn()->name() << std::endl;
    // std::cout << out.grad() << std::endl;
    out.sum().backward();

    return input.grad();
  }

  /**
   * convert a std::vector of eigen-vec to libtorch tensor
   *
   * @param vec - the vector of eigen vector to convert to torch tensor
   * @return
   */
  static torch::Tensor
  vector_of_eigen_vec_to_libtorch(const std::vector<EigenWorldspace> &vec) {
    assert(!vec.empty());

    torch::Tensor empty_tensor = torch::empty(
        {static_cast<long>(vec.size()), static_cast<long>(vec[0].size())},
        torch::TensorOptions().dtype(torch::kFloat)
        // .device(device)
        // .requires_grad(true)
    );

    // maps the empty tensor's raw pointer as an eigen matrix
    Eigen::Map<MatrixXrm<float>> empty_tensor_as_eigen(
        empty_tensor.data_ptr<float>(), empty_tensor.size(0),
        empty_tensor.size(1));

    for (int i = 0; i < vec.size(); ++i) {
      // assign the value back to the mapped data pointer
      empty_tensor_as_eigen.block(i, 0, 1, vec[0].size()) =
          vec[i].cast<float>().transpose();
    }

    return empty_tensor;
  };

public:
  torch::Device device;
  // const static torch::ScalarType scalar_type{torch::kFloat};

private:
  torch::jit::script::Module m_model;

  bool high_stream_priority;
  c10::cuda::CUDAStream cuda_stream;
  // torch::DeviceIndex device_idx;
};

/**
 * A singleton that is in charge of initialising one map per thread
 */
class TorchOccMapManager {

public:
  using TorchOccMapPtr = std::shared_ptr<TorchOccMap>;

  static TorchOccMapPtr get_occmap(int thread_id = -1) {
    // if thread_id is not given, retrieve it from omp
    if (thread_id < 0)
      thread_id = omp_get_thread_num();

    TorchOccMapManager &manager = get_instance();
    if (!manager.initialised)
      throw std::runtime_error("TorchOCcMapManager has not been initialised "
                               "yet. Run TorchOccMapManager::init() manually.");

    if (thread_id < 0 || thread_id >= manager.torch_occmap_pool.size()) {
      std::stringstream ss;
      ss << "The given thread_id '" << thread_id
         << "' is invalid for the current pool size "
         << manager.torch_occmap_pool.size();
      throw std::runtime_error(ss.str());
    }

    return manager.torch_occmap_pool[thread_id];
  }

  /**
   * Public interface to pre-initialise the manager's occ-maps
   */
  static void init() {
    TorchOccMapManager &manager = get_instance();
    manager._init();
  }

  /**
   *
   * @return the internal string reference to model path
   */
  static std::string &mutable_model_path() {
    TorchOccMapManager &manager = get_instance();
    return manager.model_path;
  }

  /**
   * For warming up cuda stream among all threads with the given `functor`.
   * Note that it is best for the functor to calls function that are actually
   * of interest, and with the same set of data size (i.e. batch size and
   * dimensionality). It is because libtorch uses JIT and if your
   * warmup_functor calls something with a different size than the actual
   * data in runtime, the JIT won't optimise the path to your data of interest.
   *
   * @param warmup_functor - a function with an argument of the shared
   *    pointer to the occ-map, called inside omp threads
   *    e.g. lambda function of:
   *    TorchOccMapManager::warmup([num_pts](TorchOccMapPtr &model) {
   *      volatile auto dummy =
   *          model->grad2({torch::rand({num_pts, 3}, model->device)});
   *    });
   * @param repeat - the number of time to repeat the warmup function
   */
  static void
  warmup(const std::function<void(TorchOccMapPtr &)> &warmup_functor,
         size_t repeat = 5) {
    // Implicitly calls init
    TorchOccMapManager &manager = get_instance();
    manager._init();
// warm up in omp threads
#pragma omp parallel default(none) shared(manager, warmup_functor, repeat)     \
    num_threads(omp_get_max_threads())
    {
      auto occmap_ptr = manager.torch_occmap_pool[omp_get_thread_num()];
      for (int _ = 0; _ < repeat; ++_)
        warmup_functor(occmap_ptr);
    }
  }

private:
  explicit TorchOccMapManager() : initialised(false) {}

  void _init() {
    if (initialised)
      return;

    ///////////////////////////////////////////
    // optimise for the best performance
    at::set_num_threads(omp_get_num_procs());
    at::globalContext().setBenchmarkCuDNN(true);

    if (model_path.empty())
      throw std::runtime_error("Model path `model_path` has not been set yet!\n"
                               "Set it with TorchOccMapManager::"
                               "mutable_model_path() = xxx;\n");
    for (int _ = 0; _ < omp_get_max_threads(); ++_) {
      torch_occmap_pool.emplace_back(std::make_shared<TorchOccMap>(model_path));
    }

    initialised = true;
  }

  static TorchOccMapManager &get_instance() {
    static TorchOccMapManager manager{};
    return manager;
  }

  std::vector<TorchOccMapPtr> torch_occmap_pool;
  std::string model_path;
  bool initialised;
};
