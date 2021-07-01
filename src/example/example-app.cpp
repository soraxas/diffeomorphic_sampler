
#include <string>

#include <iostream>
#include <memory>

#include <Eigen/Dense>

#include "cppm.hpp"
#include "soraxas_cpp_toolbox/SimpleCSVWriter.h"

#include "TorchOccMap.hpp"

#define SXS_DO_NOT_INCLUDE_FUTURE
#include "soraxas_cpp_toolbox/main.h"

int main(int argc, const char *argv[]) {

  using namespace Eigen;

  size_t batch_size = 2;

  //  Matrix<double, -1, -1> m(6, 6 * batch_size);
  MatrixXd m = Matrix<double, -1, -1>::Random(6, 6);

  sxs::println(m);
  sxs::println("m.rows() ", m.rows());
  sxs::println("m.cols() ", m.cols());

  torch::Tensor t = torch::zeros({int(batch_size), 5, 3, 6}, torch::kDouble);

  sxs::println(t);
  sxs::println("--------t");

  auto t_view = t.index({1, 2, "..."});

  Matrix<double, -1, -1, RowMajor>::Map(t_view.data_ptr<double>(),
                                        t_view.size(0), t_view.size(1)) = m;

  sxs::println(t_view);

  sxs::println("--fff------t");
  sxs::println(t);

  auto offse = torch::tensor({{-0.5, 0., 0.},
                              {+0.5, 0., 0.},
                              {0., -0.5, 0.},
                              {0., +0.5, 0.},
                              {0., 0., -0.5},
                              {0., 0., +0.5}},
                             torch::kDouble);

  sxs::println(offse);
  sxs::println("");
  sxs::println(offse.repeat({10,2,1, 1}));


  //  for (int i = 0; i < 2; ++i) {
  //    sxs::println(m.middleCols(6 * i, 6));
  //    sxs::println("");
  //  }

  return 0;

  if (argc < 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  //  omp_set_num_threads(omp_get_num_procs());
  std::cout << omp_get_max_threads() << std::endl;
  std::cout << at::get_num_threads() << std::endl;

  int num_datapts = 50;
  if (argc == 3) {
    num_datapts = std::stoi(argv[2]);
  }

  using TorchOccMapPtr = TorchOccMapManager::TorchOccMapPtr;
  TorchOccMapManager::mutable_model_path() = argv[1];
  TorchOccMapManager::warmup([num_datapts](TorchOccMapPtr &model) {
    volatile auto dummy =
        model->grad2({torch::rand({num_datapts, 3}, model->device)});
  });

  auto start_tp_all = sxs::Timer::get_timepoint();
  std::cout << omp_get_max_threads() << std::endl;

  //  for (int j = 0; j < 10; ++j)
  //#pragma omp parallel default(none) shared(std::cout, num_datapts) \
//    num_threads(omp_get_max_threads())
  {
    std::cout << "hello" << std::endl;
    // std::cout << std::thread::id << std::endl;
    TorchOccMapPtr occmap = TorchOccMapManager::get_occmap();
    std::cout << "loaded" << std::endl;

    using EigenWorldSpacePtVector = Eigen::Matrix<double, 3, 1>;

    std::cout << "---" << std::endl;
    torch::Tensor linspace = torch::linspace(-1.5, 1.5, 30);
    std::cout << linspace << std::endl;

    std::vector<EigenWorldSpacePtVector> pts;
    for (int i = 0; i < linspace.size(0); ++i) {
      for (int j = 0; j < linspace.size(0); ++j) {
        for (int k = 0; k < linspace.size(0); ++k) {
          EigenWorldSpacePtVector v;
          v[0] = linspace[i].item<double>();
          v[1] = linspace[j].item<double>();
          v[2] = linspace[k].item<double>();
          pts.emplace_back(v);
        }
      }
    }

    //    std::cout << pts << std::endl;

    auto pts_as_tensor = TorchOccMap::vector_of_eigen_vec_to_libtorch(pts);

    auto outt = occmap->grad2(pts_as_tensor.to(occmap->device));

    //    std::cout << outt << std::endl;

    auto csv_writer = CSVInstantWriter("pts.csv");

    for (int i = 0; i < pts_as_tensor.size(0); ++i) {
      csv_writer.addNewRow(pts_as_tensor.index({i, 0}).item<double>(),
                           pts_as_tensor.index({i, 1}).item<double>(),
                           pts_as_tensor.index({i, 2}).item<double>(),
                           outt.index({i, 0}).item<double>(),
                           outt.index({i, 1}).item<double>(),
                           outt.index({i, 2}).item<double>());
    }

    return 0;

    // at::AutoNonVariableTypeMode disable_autograd;
    {
      // the below is for newer version of libtorch?
      // c10::InferenceMode guard(True);
      // the below guard enables inference mode, disable autograd
      // at::AutoNonVariableTypeMode guard;

      // occmap.model().to(torch::dtype(torch::kFloat32));

      occmap->grad2(torch::rand({num_datapts, 3}).to(occmap->device));
      std::cout << 1 << std::endl;
      occmap->grad2(torch::rand({num_datapts, 3}).to(occmap->device));
      std::cout << 2 << std::endl;
      occmap->grad2(torch::rand({num_datapts, 3}).to(occmap->device));
      std::cout << 3 << std::endl;

      {
        auto timer = sxs::Timer{};

        for (int i = 0; i < 20; ++i) {

          timer.stamp("declare input");
          std::vector<torch::jit::IValue> inputs;

          std::vector<TorchOccMap::EigenWorldspace> hehe;
          hehe.resize(num_datapts);
          int cnt = 0;
          for (int i = 0; i < hehe.size(); ++i) {
            for (int j = 0; j < hehe[i].size(); ++j) {
              hehe[i][j] = double(cnt) / 100;
              cnt++;
            }
          }

          // std::cout << "=======" << std::endl;
          // std::cout << hehe << std::endl;
          // std::cout << "=======" << std::endl;

          ////////////////////////////////////////////////

          // std::cout << vector_of_eigen_vec_to_libtorch(hehe) << std::endl;
          // std::cout << "empty_tensor" << std::endl;

          timer.stamp("gen rand");
          // auto rand_num = torch::rand({num_datapts, 3});
          auto rand_num = TorchOccMap::vector_of_eigen_vec_to_libtorch(hehe);

          // std::cout << rand_num << std::endl;

          timer.stamp("put to cuda device");
          rand_num = rand_num.to(occmap->device);

          // timer.stamp("push_back to a list");
          inputs.push_back(rand_num);

          timer.stamp("forward");
          auto start = sxs::Timer::get_timepoint();
          // auto out = occmap->model().run_method("grad", rand_num);
          // auto out = occmap->model().run_method("forward", rand_num);

          // std::cout << occmap->grad2(rand_num) << std::endl;
          // std::cout << occmap->grad2(rand_num) << std::endl;
          // volatile auto ff = occmap->grad2(rand_num);
          auto ff = occmap->grad2(rand_num);
          timer.stamp("forward done");
          // std::cout << occmap->grad2(rand_num) << std::endl;

          std::cout << "ff" << std::endl;
          std::cout << ff << std::endl;
          std::cout << "ff" << std::endl;

          std::vector<long> vvv = {0, 1, 3, 5, 6, 8};

          auto tt =
              torch::from_blob(vvv.data(), vvv.size(),
                               torch::TensorOptions().dtype(torch::kLong));

          std::cout << tt << std::endl;
          std::cout << "tt" << std::endl;

          std::cout << "---------tt-----------" << std::endl;
          std::cout << ff.index({tt, torch::indexing::Slice()}) << std::endl;
          std::cout << ff.index({tt, torch::indexing::Slice()}).mean(0, true)
                    << std::endl;
          std::cout << ff.index({tt, torch::indexing::Slice()})
                           .mean(0, true)
                           .to(torch::kDouble)
                    << std::endl;

          torch::Tensor consolidated_grad =
              ff.index({tt, torch::indexing::Slice()}).mean(0);

          std::vector<EigenWorldSpacePtVector> vec;

          vec.emplace_back(Eigen::Map<EigenWorldSpacePtVector>(
              consolidated_grad.to(torch::kCPU, torch::kDouble)
                  .contiguous()
                  .data_ptr<double>(),
              consolidated_grad.size(0)));

          std::cout << vec << std::endl;
          std::cout << vec << std::endl;
          std::cout << "---" << std::endl;
          std::cout << rand_num.repeat({10, 1}) << std::endl;

          std::cout << "---" << std::endl;
          std::cout << rand_num.index({rand_num.pow(2).sum(1).argmax(),
                                       torch::indexing::Slice()})
                    << std::endl;

          //          auto ff = occmap->grad2(rand_num);

          //          auto selected_tensor = ff.index({tt,
          //          torch::indexing::Slice()}).sum
          //                                 (1);
          //
          ////          for (int i = 0; i < ff.size(0); ++i) {
          //            std::cout << selected_tensor.square().sum()
          //                      << std::endl;
          //            std::cout <<
          //            selected_tensor.index({selected_tensor.square().sum
          //                             (), torch::indexing::Slice()})
          //                      << std::endl;
          //          }

          std::cout << sxs::Timer::timepoint_diff_to_second(
                           sxs::Timer::get_timepoint() - start)
                    << std::endl;
          // auto out = occmap->model().forward(inputs);
          timer.stamp("retrieve results");

          /*

          //////////////////////////////
          auto out_tensor = out.toTensor().to(torch::kCPU).squeeze(1);
          // std::cout << out_tensor << std::endl;











          Eigen::Map<MatrixXrm<float>>
            E(out_tensor.data_ptr<float>(), out_tensor.size(0),
          out_tensor.size(1));
          // std::cout << E << std::endl;
          // std::cout << E.cols() << std::endl;
          // std::cout << E.rows() << std::endl;
          //////////////////////////////

          using EigenWorldspace = Eigen::Matrix<double, 3, 1>;
          std::vector<EigenWorldspace> vecs;
          vecs.reserve(num_datapts);
          for (int i = 0; i < out_tensor.size(0); ++i)
            vecs.emplace_back(E.block(i, 0, 1, 3).transpose().cast<double>());

          // std::cout << vecs << std::endl;
          // break;

          */

          timer.stamp("rest");
        }
      }
    }
  }

  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////

  //  auto csv_writer = CSVWriterStream(3);
  auto csv_writer = CSVInstantWriter("occmap-bench.csv");
  {
    TorchOccMapPtr occmap = TorchOccMapManager::get_occmap();
    at::AutoNonVariableTypeMode guard;

    for (int num_threads : cppm::range(11)) {
      omp_set_num_threads(num_threads);

      std::vector<int> bench_pts = {5,    10,   20,   50,   100,  200,
                                    500,  1000, 1500, 2000, 2500, 3000,
                                    4000, 5000, 7500, 10000};
      for (int num_datapts : cppm::iter(bench_pts)) {
        std::vector<TorchOccMap::EigenWorldspace> input_val;
        input_val.resize(num_datapts);
        int cnt = 0;
        for (int i = 0; i < input_val.size(); ++i) {
          for (int j = 0; j < input_val[i].size(); ++j) {
            input_val[i][j] = double(cnt) / 100;
            cnt++;
          }
        }

        // eigen vec to tensor
        auto rand_num = TorchOccMap::vector_of_eigen_vec_to_libtorch(input_val);
        rand_num = rand_num.to(occmap->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(rand_num);

        // warm up
        for (int i = 0; i < 20; ++i) {
          volatile auto ff = occmap->grad2(rand_num);
        }

        // time
        for (int i = 0; i < 20; ++i) {
          auto start = sxs::Timer::get_timepoint();

          volatile auto ff = occmap->grad2(rand_num);

          double time = sxs::Timer::timepoint_diff_to_second(start);
          //        csv_writer << num_datapts << time;
          csv_writer.addNewRow(num_threads, num_datapts, time);
        }
      }
    }
  }

  //  csv_writer.writeToFile("occmap-bench.csv");

  std::cout << "ok\n";
  std::cout << sxs::Timer::timepoint_diff_to_second(start_tp_all) << "\n";
}

// int main(int argc, const char* argv[]) {
//   if (argc < 2) {
//     std::cerr << "usage: example-app <path-to-exported-script-module>\n";
//     return -1;
//   }

//   at::init_num_threads();

//   int num_datapts = 50;
//   if (argc == 3) {
//     num_datapts = std::stoi(argv[2]);
//   }

//   torch::Device device = torch::kCPU;
//   std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() <<
//   std::endl; if (torch::cuda::is_available())
//   {
//     std::cout << "CUDA is available! Operating on GPU." << std::endl;
//     device = torch::kCUDA;
//   }

//   torch::jit::script::Module module;

//   try {
//     // Deserialize the ScriptModule from a file using torch::jit::load().
//     module = torch::jit::load(argv[1]);
//   }
//   catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return -1;
//   }

//   at::AutoNonVariableTypeMode disable_autograd;
//   at::globalContext().setBenchmarkCuDNN(true);
//   {
//     // the below is for newer version of libtorch?
//     // c10::InferenceMode guard(True);
//     // the below guard enables inference mode, disable autograd
//     // at::AutoNonVariableTypeMode guard;

//     module.to(device, torch::kFloat32);
//     // module.to(torch::dtype(torch::kFloat32));

//     {
//       auto timer = sxs::Timer{};

//       for (int i = 0; i < 20; ++i){

//         timer.stamp("declare input");
//         std::vector<torch::jit::IValue> inputs;

//         // inputs.push_back(torch::ones({1, 3, 224, 224}).to(device));
//         // inputs.push_back(torch::rand({300, 1, 1200}).to(device));

//         // std::vector<Eigen::Matrix<float>

//         // torch::from_blob(data, {num_datapts, 3})

//         using EigenWorldspace = Eigen::Matrix<double, 3, 1>;
//         std::vector<EigenWorldspace> hehe;
//         hehe.resize(num_datapts);
//         int cnt = 0;
//         for (int i = 0; i < hehe.size(); ++i){
//           for (int j = 0; j < hehe[i].size(); ++j){
//             hehe[i][j] = cnt;
//             cnt ++;
//           }
//         }

//         std::cout << "=======" << std::endl;
//         std::cout << hehe << std::endl;
//         std::cout << "=======" << std::endl;

//         auto vector_of_eigen_vec_to_libtorch = [](const
//         std::vector<EigenWorldspace>&vec){
//         // std::cout << vec[0].rows() << std::endl;
//         // std::cout << vec[0].cols() << std::endl;
//           torch::Tensor empty_tensor = torch::empty({vec.size(),
//           vec[0].size()}, torch::dtype(torch::kFloat32));

//           Eigen::Map<MatrixXrm<float>>
//             empty_tensor_as_eigen(empty_tensor.data_ptr<float>(),
//             empty_tensor.size(0), empty_tensor.size(1));
//           for (int i = 0; i < vec.size(); ++i){
//             empty_tensor_as_eigen.block(i, 0, 1, vec[0].size()) =
//             vec[i].cast<float>().transpose();
//           }

//           return empty_tensor;
//         };

//         std::cout << vector_of_eigen_vec_to_libtorch(hehe) << std::endl;
//         std::cout << "empty_tensor" << std::endl;

//         timer.stamp("gen rand");
//         // auto rand_num = torch::rand({num_datapts, 3});
//         auto rand_num = vector_of_eigen_vec_to_libtorch(hehe);

//         timer.stamp("put to cuda device");
//         rand_num = rand_num.to(device);

//         // timer.stamp("push_back to a list");
//         inputs.push_back(rand_num);

//         timer.stamp("forward");
//         auto out = module.run_method("grad", rand_num);
//         // auto out = module.forward(inputs);

//         //////////////////////////////
//         auto out_tensor = out.toTensor().to(torch::kCPU).squeeze(1);
//         std::cout << out_tensor << std::endl;
//         // std::cout << out_tensor.size(1) << std::endl;

//         Eigen::Map<MatrixXrm<float>>
//           E(out_tensor.data_ptr<float>(), out_tensor.size(0),
//           out_tensor.size(1));
//         // std::cout << out_tensor.size(2) << std::endl;
//         std::cout << E << std::endl;
//         std::cout << E.cols() << std::endl;
//         std::cout << E.rows() << std::endl;
//         //////////////////////////////

//         using EigenWorldspace = Eigen::Matrix<double, 3, 1>;
//         std::vector<EigenWorldspace> vecs;
//         vecs.reserve(num_datapts);
//         for (int i = 0; i < out_tensor.size(0); ++i)
//           vecs.emplace_back(E.block(i, 0, 1, 3).transpose().cast<double>());

//         std::cout << vecs << std::endl;
//         break;

//         timer.stamp("rest");
//         }
//     }
//   }

//   std::cout << "ok\n";
// }
