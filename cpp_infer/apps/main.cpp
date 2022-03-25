#include <ocr_rec.h>
int main(int argc, char **argv) {

    PaddleOCR::CRNNRecognizer rec("/paddle/model", true, 0,
                       200, 1,
                       1, "/paddle/model/plate/dict.txt",
                       0, "fp32", 6);

    std::vector<cv::Mat> img_list;
    cv::Mat srcimg = cv::imread("/paddle/cpp_infer/imgs/plate.jpg", cv::IMREAD_COLOR);
    img_list.push_back(srcimg);
    std::cout<<"Hehe\n"<<std::endl;

    std::vector<double> rec_times;
    rec.Run(img_list, &rec_times);
    return 0;
}