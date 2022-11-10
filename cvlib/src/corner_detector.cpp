/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>
#include <random>

#include <ctime>


    std::vector<cv::Point> fast_offsets = {
        cv::Point(0, 0), - // zero point (not need)
        cv::Point(0, -3), cv::Point(1, -3),  cv::Point(2, -2), // 1, 2, 3
        cv::Point(3, -1), cv::Point(3, 0),   cv::Point(3, 1),  // 4, 5, 6
        cv::Point(2, 2),  cv::Point(1, 3),   cv::Point(0, 3),  // 7, 8, 9
        cv::Point(-1, 3), cv::Point(-2, 2),  cv::Point(-3, 1), // 10, 11, 12
        cv::Point(-3, 0), cv::Point(-3, -1), cv::Point(-2, -2), cv::Point(-1, -3) // 13, 14, 15, 16
     };

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    keypoints.clear();
    if ((image.rows() < 7) || (image.cols() < 7))
        return;
    int t = 10; //threshold for fast algorithm
    cv::Mat img_mat = image.getMat();
    // \todo implement FAST with minimal LOCs(lines of code), but keep code readable.
    cv::blur(img_mat, img_mat, cv::Size(5, 5)); //Дополнительно добавлено гауссово сглаживания для устранения шумов
    for (int i = 3; i < image.rows() - 3; i++)
        for (int j = 3; j < image.cols() - 3; j++)
        {
            int compare_arr[17] = {0}; // compare result array (-1 - darker, 0 - same, 1 - lighter)
            cv::Point pix_pos = cv::Point(j, i);
            int Ip = img_mat.at<uint8_t>(pix_pos); // suspect pixel
            //check pixels 1, 5, 9, 13
            int count_l = 0, count_d = 0; //counters for lighter and darker pixels
            for (int k = 1; k <= 13; k += 4)
            {
                int Ipi = img_mat.at<uint8_t>(pix_pos + fast_offsets[k]);
                if (Ipi > Ip + t)
                {
                    compare_arr[k] = 1;
                    count_l++;
                }
                else if (Ipi < Ip - t)
                {
                    compare_arr[k] = -1;
                    count_d++;
                }
                else
                    compare_arr[k] = 0;
            }
            if ((count_l >= 3) || (count_d >= 3)) 
            {
                // The pixels have been verified. Check the remaining pixels
                for (int k = 1; k <= 16; k++)
                {
                    if ((k == 1) || (k == 5) || (k == 9) || (k == 13))
                        continue;
                    int Ipi = img_mat.at<uint8_t>(pix_pos + fast_offsets[k]);
                    if (Ipi > Ip + t)
                        compare_arr[k] = 1;
                    else if (Ipi < Ip - t)
                        compare_arr[k] = -1;
                    else
                        compare_arr[k] = 0;
                }
                if (check_count_same_in_a_row(compare_arr + 1, 12))
                    keypoints.push_back(cv::KeyPoint(pix_pos, 10));
            }
        }
}

bool corner_detector_fast::check_count_same_in_a_row(int* arr, int thresh)
{
    int* first = arr;
    int* last = arr + 15;
    int* prev = first;
    int count = 1;
    int max_count = 1;

    while (++first != last)
    {
        if (*(prev++) == *first)
            ++count;
        else
        {
            count = 1;
        }
        if (count > max_count)
            max_count = count;
    }
    
    if (max_count >= thresh)
        return (true);

    return (false);
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    /*
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
    */
    // std::cout << "desc_mat" << desc_mat << std::endl;

    int patch_size = 31; // neighborhood of keypoint (patch_size * patch_size)

    const int desc_length = 32; // 256 compares => 256 bits => 32 bytes => 32 uint8_t
    cv::Mat temp_desc_mat = cv::Mat(static_cast<int>(keypoints.size()), desc_length, CV_8U);

    //auto img = image.getMat();
    cv::Mat img;
    image.getMat().copyTo(img);
    temp_desc_mat.setTo(0);

    cv::GaussianBlur(img, img, cv::Size(5, 5), 2, 2); //Размытие по гаусу
    if (_pairs_offset.empty())
    {
        create_random_pairs(256, patch_size);
    }
    

    uint8_t* ptr = reinterpret_cast<uint8_t*>(temp_desc_mat.ptr());
    int skiped_keypoints = 0;
    for (const auto& pt : keypoints)
    {
        // keypoints on the edges of the image are skipped
        if ((pt.pt.x < patch_size / 2) || (pt.pt.y < patch_size / 2))
        {
            skiped_keypoints++;
            continue;
        }

        if ((pt.pt.x > (img.cols - patch_size / 2)) || (pt.pt.y > (img.rows - patch_size / 2)))
        {
            skiped_keypoints++;
            continue;
        }

        int idx = 0;
        for (int i = 0; i < desc_length; ++i)
        {
            uint8_t temp = 0;
            for (int j = 0; j < 8; j++)
            {
                temp |= (binary_test(img, pt.pt, _pairs_offset[idx]) << (7 - j));
                idx++;
            }
            *ptr = temp;
            ++ptr;
        }
    }

    // copy temp descriptor mat to output array without zero rows (because skipping keypoints on the edges of the image)
    descriptors.create(static_cast<int>(keypoints.size() - skiped_keypoints), desc_length, CV_8UC1);
    auto desc_mat = descriptors.getMat();
    temp_desc_mat(cv::Range(0, temp_desc_mat.rows - skiped_keypoints), cv::Range(0, temp_desc_mat.cols)).copyTo(desc_mat);
    descriptors.getMat() = desc_mat;
    //std::cout << " descriptors.getMat()" <<  descriptors.getMat() << std::endl;
}


void corner_detector_fast::create_random_pairs(int pairs_count, int patch_size)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, patch_size/2/3);
    int x1, y1, x2, y2;
    int max_abs_x = patch_size / 2;
    int max_abs_y = patch_size / 2;

    for (int i = 0; i < pairs_count; i++)
    {
        x1 = (int)distribution(generator);
        y1 = (int)distribution(generator);
        x2 = (int)distribution(generator);
        y2 = (int)distribution(generator);

        if (std::abs(x1) > max_abs_x) x1 = (x1 < 0 ? -max_abs_x : max_abs_x);
        if (std::abs(y1) > max_abs_y) y1 = (y1 < 0 ? -max_abs_y : max_abs_y);
        if (std::abs(x2) > max_abs_x) x2 = (x2 < 0 ? -max_abs_x : max_abs_x);
        if (std::abs(y2) > max_abs_y) y2 = (y2 < 0 ? -max_abs_y : max_abs_y);
        
        _pairs_offset.push_back(pair(cv::Point(x1,y1), cv::Point(x2,y2)));
    }
}

uint8_t corner_detector_fast::binary_test(cv::Mat image, cv::Point keypoint, pair p)
{
    if (image.at<uint8_t>(keypoint + p.offset1) < image.at<uint8_t>(keypoint + p.offset2))
        return 1;
    else
        return 0;
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
