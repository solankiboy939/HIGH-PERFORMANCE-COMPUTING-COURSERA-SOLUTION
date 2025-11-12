#include <opencv2/opencv.hpp>
using namespace cv;
int main() {
    Mat img(64, 64, CV_8UC3, Scalar(50, 100, 150)); // small blue-ish image
    imwrite("sloth.png", img);
    return 0;
}
