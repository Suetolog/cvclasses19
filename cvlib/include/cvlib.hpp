/* Computer Vision Functions.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \return segmented image
cv::Mat split_and_merge(const cv::Mat& image, double stddev, int mihWidth, int minHeight);
//cv::Mat split_and_merge(const cv::Mat& image, double stddev);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm

class gaussProcess
{
    public:
    gaussProcess();
    gaussProcess(const cv::Mat& Img, const int& img_type, const cv::Size& img_size, const int& std);
    ~gaussProcess();

    void update_background_mask(int thresh);
    cv::Mat get_background_mask(void);
    cv::Mat get_weights(void);
    cv::Mat check_thresh(int thresh);
    void set_image(cv::Mat& img);
    void update_statistics(double alph1, double alph2);
    void rewrite_statistics(cv::Mat mask, int std);
    void update_weights(uint8_t curr_K, double alph3, cv::Mat input_mask, bool is_current_process_flag);
    //void do_classification(cv::Mat& bgmask, cv::Mat weight_mask, double weight_thresh);

    private:
    cv::Mat _weights;
    cv::Mat _means;
    // cv::Mat _std2;
    cv::Mat _std;
    cv::Mat _background_mask;

    cv::Size _img_size;
    int _img_type;
    cv::Mat _img;
};

class motion_segmentation : public cv::BackgroundSubtractor
{
    public:
    /// \brief ctor
    motion_segmentation(int threshold, double alpha1, double alpha2, double alpha3, double weight_thresh, uint8_t K, int init_std);

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = -1) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override
    {
        backgroundImage.assign(bg_model_);
    }

    void setVarThreshold(int threshold);

    private:
    cv::Mat bg_model_;

    cv::Mat _fgmask;
    cv::Mat _bgmask;

    bool _first_frame;
    cv::Size _img_size;
    int _img_type;

    uint8_t _max_K;
    uint8_t _curr_K;

    int _init_std;
    int _thresh;
    double _alpha1;
    double _alpha2;
    double _alpha3;
    double _weight_thresh;

    std::map<int, gaussProcess> _models;
};

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
    public:
    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    /// \see Feature2d::compute
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;

    /// \see Feature2d::getDefaultName
    virtual cv::String getDefaultName() const override
    {
        return "FAST_Binary";
    }

    //if the number of consecutive same elements of the arr array is greater than or equal to thresh, func returns true
    //arr is a mass with size 17, where arr[0] dont use 
    bool check_count_same_in_a_row(int* arr, int thresh); //
    typedef struct pair
    {
        cv::Point offset1;
        cv::Point offset2;

        pair(cv::Point of1, cv::Point of2)
        {
            offset1 = of1;
            offset2 = of2;
        }
    };
    //this function creates "pairs_count" random pairs (offsets from keypoint) in patch with size (patch_size * patch_size)
    void create_random_pairs(int pairs_count, int patch_size);
    //this function performs a comparison of two pixels 
    //it returns 1 if the intensity of the pixel of the first offset from keypoint is less than the second, else 0
    uint8_t binary_test(cv::Mat image, cv::Point keypoint, pair p);

    private:
    std::vector<pair> _pairs_offset;
};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    /// \brief ctor
    descriptor_matcher(int ratio = 1) : ratio_(ratio)
    {
    }

    /// \brief setup ratio threshold for SSD filtering
    void set_ratio(float r)
    {
        ratio_ = r;
    }

    protected:
    /// \see cv::DescriptorMatcher::knnMatchImpl
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::radiusMatchImpl
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual bool isMaskSupported() const override
    {
        return false;
    }

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override
    {
        cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
        if (emptyTrainData)
        {
            copy->clear();
        }
        return copy;
    }

    private:
    int calc_hamming_distance(uint8_t* desc1, uint8_t* desc2, uint8_t size); //size - descriptors size in bytes

    int ratio_;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
public:
    Stitcher();
    ~Stitcher(){}
        
    void init(cv::Mat init_image, int mather_ratio = 15);
    void stitch(cv::Mat input_image);
    cv::Mat get_panoram_image(void);

    private:
    cv::Mat _panoram;
    std::vector<cv::KeyPoint> _panoram_corners;
    cv::Mat _panoram_descriptors;
    //cv::Ptr<corner_detector_fast> _corn_detector;
    cv::Ptr<cv::AKAZE> _corn_detector;
    descriptor_matcher _decr_matcher;

};
} // namespace cvlib

#endif // __CVLIB_HPP__
