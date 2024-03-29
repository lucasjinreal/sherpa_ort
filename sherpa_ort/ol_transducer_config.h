// sherpa-onnx/csrc/online-transducer-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_

#include <string>
#include <sstream>


namespace sherpa_onnx {

struct OnlineTransducerModelConfig {
  std::string encoder_filename;
  std::string decoder_filename;
  std::string joiner_filename;
  int32_t num_threads;
  bool debug = false;

  OnlineTransducerModelConfig() = default;
  OnlineTransducerModelConfig(const std::string &encoder_filename,
                              const std::string &decoder_filename,
                              const std::string &joiner_filename,
                              int32_t num_threads, bool debug)
      : encoder_filename(encoder_filename), decoder_filename(decoder_filename),
        joiner_filename(joiner_filename), num_threads(num_threads),
        debug(debug) {}

  std::string ToString() const {
    std::ostringstream os;

    os << "OnlineTransducerModelConfig(";
    os << "encoder_filename=\"" << encoder_filename << "\", ";
    os << "decoder_filename=\"" << decoder_filename << "\", ";
    os << "joiner_filename=\"" << joiner_filename << "\", ";
    os << "num_threads=" << num_threads << ", ";
    os << "debug=" << (debug ? "True" : "False") << ")";

    return os.str();
  }
};

} // namespace sherpa_onnx

#endif // SHERPA_ONNX_CSRC_ONLINE_TRANSDUCER_MODEL_CONFIG_H_
