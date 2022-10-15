file(GLOB SRCS *.cc *h)

add_executable(sherpa-onnx main.cpp ${SRCS})
target_link_libraries(sherpa-onnx onnxruntime kaldiio)
