#ifndef IMAGE_PROCESS_PART_H
#define IMAGE_PROCESS_PART_H

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/imgproc/imgproc.hpp"

void preprocess(cv::Mat &bgr, cv::Mat &gray);
cv::Point quick_find_center(const cv::Mat &gray);
void get_sobel(const cv::Mat &gray, cv::Mat &dst);
void dynamic_erode(cv::Mat &bin, cv::Point center, double dt=M_PI/180);
void dynamic_dilate(cv::Mat &bin, cv::Point center, double dt=M_PI/180);
cv::RotatedRect get_rbox(cv::Mat &im);
void elliptical_integrate(const cv::Mat &gray, cv::RotatedRect &el, std::vector<double> &avg);

#endif
