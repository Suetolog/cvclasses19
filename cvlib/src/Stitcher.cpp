
#include "cvlib.hpp"
#include <cmath> 

namespace cvlib
{
Stitcher::Stitcher()
{
    //_panoram = cv::Mat(0, 0, CV_8U);
}

void Stitcher::init(cv::Mat init_image, int mather_ratio)
{
    init_image.copyTo(_panoram);
	//_corn_detector = cvlib::corner_detector_fast::create();
    _corn_detector = cv::AKAZE::create();
    _decr_matcher = cvlib::descriptor_matcher(mather_ratio);

    //_corn_detector->detectAndCompute(_panoram, cv::Mat(), _panoram_corners, _panoram_descriptors);
}

void Stitcher::stitch(cv::Mat input_image)
{
    _corn_detector->detectAndCompute(_panoram, cv::Mat(), _panoram_corners, _panoram_descriptors);

    std::vector<cv::KeyPoint> input_img_corners;
    cv::Mat input_img_decriptors;

    _corn_detector->detectAndCompute(input_image, cv::Mat(), input_img_corners, input_img_decriptors);

    std::vector<std::vector<cv::DMatch>> matches;
    _decr_matcher.radiusMatch(input_img_decriptors, _panoram_descriptors, matches, 100.0f);

    std::vector<cv::Point2f> key_points_panoram, key_points_input;
    //Получаем не пустые ключевые матчнутые точки
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (!matches[i].empty())
        {
            key_points_input.push_back(input_img_corners[matches[i][0].queryIdx].pt);
            key_points_panoram.push_back(_panoram_corners[matches[i][0].trainIdx].pt);
        }
        
    }
    if ((key_points_input.size() < 4) || (key_points_panoram.size() < 4))
    {
        // TODO:error! not enought keypoints for homography
        return;
    }
    cv::Mat homog = cv::findHomography(cv::Mat(key_points_input), cv::Mat(key_points_panoram), cv::RANSAC);
    //for debug
    //std::cout << "homog = "<< homog << std::endl;
    //std::cout << "homog type = " << homog.type() << std::endl;

    float f_x_offset = homog.at<double>(2);
    int i_x_offset_abs = (int)ceil(abs(f_x_offset));

    //std::cout << "x_offset = " << i_x_offset_abs << std::endl;

    int new_panoram_img_width = _panoram.cols + i_x_offset_abs;
    int new_panoram_img_height = _panoram.rows; // + (int)abs(ceil(homog.at<double>(5)));
    cv::Size new_panoram_size = cv::Size(new_panoram_img_width, new_panoram_img_height);
    cv::Mat new_panoram = cv::Mat(new_panoram_size, CV_8U);
    cv::warpPerspective(input_image, new_panoram, homog, new_panoram_size, cv::INTER_CUBIC);
    cv::Mat roi;

    roi = cv::Mat(new_panoram, cv::Rect(0, 0, _panoram.cols, _panoram.rows));
    _panoram.copyTo(roi);

    new_panoram.copyTo(_panoram);
    
}

cv::Mat Stitcher::get_panoram_image(void)
{
    return (_panoram);
}
} // namespace cvlib