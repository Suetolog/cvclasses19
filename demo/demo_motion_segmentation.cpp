/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    auto mseg = cvlib::motion_segmentation(5, 0.2, 0.2, 0.4, 1.1, 2, 4); // \todo use cvlib::motion_segmentation 
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    int threshold = 5;
    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::createTrackbar("th", demo_wnd, &threshold, 255);

    cv::Mat frame;
    cv::Mat frame_mseg;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        cv::blur(frame, frame, cv::Size(5, 5)); //ƒополнительно добавлено гауссово сглаживани€ дл€ устранени€ шумов и более качетсвенной сегментации
        cv::imshow(main_wnd, frame);

        mseg.setVarThreshold(threshold); // \todo use TackbarCallback
        mseg.apply(frame, frame_mseg);
        if (!frame_mseg.empty())
            cv::imshow(demo_wnd, frame_mseg);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
