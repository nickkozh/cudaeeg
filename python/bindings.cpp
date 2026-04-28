#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <string>

#include "../include/cuda_eeg_prep.h"

namespace py = pybind11;

namespace {

Band band_from_str(const std::string& s) {
    if (s == "broadband") return BAND_BROADBAND;
    if (s == "delta")     return BAND_DELTA;
    if (s == "theta")     return BAND_THETA;
    if (s == "alpha")     return BAND_ALPHA;
    if (s == "beta")      return BAND_BETA;
    if (s == "gamma")     return BAND_GAMMA;
    if (s == "ssvep")     return BAND_SSVEP;
    throw std::invalid_argument("unknown band: " + s);
}

class Pipeline {
public:
    Pipeline(int n_ch, int win, float fs) : n_ch_(n_ch), win_(win) {
        p_ = eeg_create(n_ch, win, fs);
        if (!p_) throw std::runtime_error("eeg_create failed");
    }
    ~Pipeline() { eeg_destroy(p_); }
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    py::array_t<float> process(py::array_t<float, py::array::c_style | py::array::forcecast> raw) {
        if (raw.ndim() != 2 || raw.shape(0) != n_ch_ || raw.shape(1) != win_) {
            throw std::invalid_argument("raw must have shape (n_ch, win)");
        }
        py::array_t<float> out({(py::ssize_t)n_ch_, (py::ssize_t)EEG_N_BANDS});
        eeg_process_window(p_, raw.data(), out.mutable_data());
        return out;
    }

private:
    CudaEegPipeline* p_ = nullptr;
    int n_ch_, win_;
};

py::array_t<float> filter_band(const std::string& band,
                               py::array_t<float, py::array::c_style | py::array::forcecast> sig)
{
    if (sig.ndim() != 2) throw std::invalid_argument("signal must be 2D (n_ch, n_samp)");
    int n_ch   = (int)sig.shape(0);
    int n_samp = (int)sig.shape(1);
    py::array_t<float> out({(py::ssize_t)n_ch, (py::ssize_t)n_samp});
    eeg_filter_band(band_from_str(band), sig.data(), out.mutable_data(), n_ch, n_samp);
    return out;
}

} // anon

PYBIND11_MODULE(cuda_eeg_prep, m) {
    m.doc() = "CUDA-accelerated real-time EEG preprocessing for Jetson Orin Nano";

    py::class_<Pipeline>(m, "Pipeline")
        .def(py::init<int, int, float>(),
             py::arg("n_ch") = 8, py::arg("win") = 256, py::arg("fs") = 250.0f,
             "Create a streaming pipeline. Bandpass + notch + CAR + Welch PSD.")
        .def("process", &Pipeline::process,
             "Process one window. raw: (n_ch, win) float32 → returns (n_ch, 6) float32 band powers.");

    m.def("filter_band", &filter_band,
          py::arg("band"), py::arg("signal"),
          "Standalone narrowband FIR. band ∈ {delta,theta,alpha,beta,gamma,ssvep,broadband}. "
          "signal: (n_ch, n_samp) float32 → returns (n_ch, n_samp) float32.");

    m.attr("N_BANDS") = py::int_(EEG_N_BANDS);
}
