/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.clear();
    matches.resize(q_desc.rows);
    //cv::RNG rnd;

    int min_distance;
    int distance;
    int t_row_min;
    bool find_flag;
    uint8_t* q_desc_ptr;
    uint8_t* t_desc_ptr;

    for (int i = 0; i < q_desc.rows; ++i)
    {
        min_distance = ratio_;
        t_row_min = 0;
        find_flag = false;
        for (int j = 0; j < t_desc.rows; ++j)
        {
            q_desc_ptr = q_desc.ptr(i, 0);
            t_desc_ptr = t_desc.ptr(j, 0);
            distance = calc_hamming_distance(q_desc_ptr, t_desc_ptr, 32);
            if (distance < min_distance)
            {
                min_distance = distance;
                t_row_min = j;
                find_flag = true;
            }
        }
        if (find_flag)
          {
              matches[i].emplace_back(i, t_row_min, min_distance);
          }
        // \todo implement Ratio of SSD check.

       // matches[i].emplace_back(i, rnd.uniform(0, t_desc.rows), FLT_MAX);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float /*maxDistance*/,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    // \todo implement matching with "maxDistance"
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}

int descriptor_matcher::calc_hamming_distance(uint8_t* desc1, uint8_t* desc2, uint8_t size)
{
    int res = 0;
    uint8_t tmp;

    for (auto i = 0; i < size; i++)
    {
        tmp = *(desc1 + i) ^ *(desc2 + i);
        while (tmp)
        {
            ++res;
            tmp &= tmp - 1;
        }
    }
    return (res);
}
} // namespace cvlib
