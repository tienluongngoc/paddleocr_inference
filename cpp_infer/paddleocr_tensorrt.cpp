#include <iostream>
#include <chrono>
#include <map>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 100;
static const int OUTPUT_SIZE = 25 * 38;
const char* INPUT_BLOB_NAME = "x";
const char* OUTPUT_BLOB_NAME = "save_infer_model/scale_0.tmp_1";
static Logger gLogger;

const int ks[] = {3, 3, 3, 3, 3, 3, 2};
const int ps[] = {1, 1, 1, 1, 1, 1, 0};
const int ss[] = {1, 1, 1, 1, 1, 1, 1};
const int nm[] = {64, 128, 256, 256, 512, 512, 512};
const std::string alphabet = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
                             

using namespace nvinfer1;

std::string strDecode(std::vector<int>& preds, bool raw) {
    std::string str;
    if (raw) {
        for (auto v: preds) {
            str.push_back(alphabet[v]);
        }
    } else {
        for (size_t i = 0; i < preds.size(); i++) {
            if (preds[i] == 0 || (i > 0 && preds[i - 1] == preds[i])) continue;
            str.push_back(alphabet[preds[i]]);
        }
    }
    return str;
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    char *trtModelStream{nullptr};
    size_t size{0};

    std::ifstream file("/test/model/paddle.engine", std::ios::binary);
    if (file.good()) 
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }
    
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    void* buffers[2];

    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];

    cv::Mat img = cv::imread("/test/images/08.jpg");
    if (img.empty()) {
        std::cerr << "demo.png not found !!!" << std::endl;
        return -1;
    }
    cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (uc_pixel[2] / 255.0-0.5)/0.5;
            data[i + INPUT_H * INPUT_W] = (uc_pixel[1] / 255.0-0.5)/0.5;
            data[i + 2 * INPUT_H * INPUT_W] = (uc_pixel[0] / 255.0-0.5)/0.5;
            uc_pixel += 3;
            ++i;
        }
    }

    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<int> preds;
    for (int i = 0; i < 25; i++) {
        int maxj = 0;
        for (int j = 1; j < 38; j++) {
            if (prob[38 * i + j] > prob[38 * i + maxj]) maxj = j;
        }
        preds.push_back(maxj);
    }
    std::cout << "sim: " << strDecode(preds, false) << std::endl;

    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}