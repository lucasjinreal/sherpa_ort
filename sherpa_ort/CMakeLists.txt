add_executable(online-fbank-test online-fbank-test.cc)
target_link_libraries(online-fbank-test kaldiio)

add_executable(sherpa-onnx main.cpp)
target_link_libraries(sherpa-onnx onnxruntime kaldiio)
