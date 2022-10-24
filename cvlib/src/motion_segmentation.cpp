/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include <cvlib.hpp>

#include <iostream>
#include <math.h>

namespace cvlib
{
    /* gaussProcess Functions */
    gaussProcess::gaussProcess()
    {
        _weights = 0;
        _means = 0;
        //_std2 = 0;
        _std = 0;
        _background_mask = 0;
        _img_size = cv::Size(0,0);
        _img_type = 0;
    }

    gaussProcess::gaussProcess(const cv::Mat &Img, const int &img_type, const cv::Size &img_size, const int &std) // хз на счет double
    {
        _img_size = img_size;
        _img_type = img_type;
        _weights = cv::Mat::ones(_img_size, CV_32F);
        _means = Img;
        _std = cv::Mat(_img_size, _img_type, cv::Scalar(std));
        //_std2 = cv::Mat(_img_size, _img_type, cv::Scalar(pow(std,2)));
        _background_mask = cv::Mat::zeros(_img_size, _img_type);
    }

    gaussProcess::~gaussProcess(){}

    void gaussProcess::update_background_mask(int thresh)
    {
        _background_mask = check_thresh(thresh);
    }

    cv::Mat gaussProcess::get_background_mask(void)
    {
        return (this->_background_mask);
    }

    cv::Mat gaussProcess::get_weights(void)
    {
        return (this->_weights);
    }

    cv::Mat gaussProcess::check_thresh(int thresh)
    {
        return ((cv::abs(_means - _img) / _std) <= thresh);
    }

    void gaussProcess::set_image(cv::Mat &img)
    {
        _img = img;
    }

    void gaussProcess::update_statistics(double alph1, double alph2)
    {
        cv::Mat fg_msk(_img_size, _img_type);
        cv::bitwise_not(_background_mask, fg_msk);
        cv::Mat msk = _background_mask / 255 ;
        fg_msk = fg_msk / 255;

        _means = ((1 - alph1) * _means.mul(msk)) + (_means.mul(fg_msk)) + (alph1 * _img.mul(msk));
        _std = ((1 - alph2) * _std.mul(msk)) + (_std.mul(fg_msk)) + alph2 * (cv::abs(_img - _means)).mul(msk);
        //std::cout << "_std = " << std::endl << _std << std::endl;
    }

    void gaussProcess::rewrite_statistics(cv::Mat mask, int std)
    {
        _img.copyTo(_means, mask);
        _std.setTo(std, mask);
    }

    void gaussProcess::update_weights(uint8_t curr_K, double alph3, cv::Mat input_mask, bool is_current_process_flag) 
    {
        cv::Mat mask(input_mask.size(), input_mask.type());
        cv::Mat not_msk(input_mask.size(), input_mask.type());
        input_mask.copyTo(mask);
        cv::bitwise_not(mask, not_msk);
        mask = mask / 255; // bool mask {0, 255} -> {0, 1}
        not_msk = not_msk / 255;
        mask.convertTo(mask, CV_32F);
        not_msk.convertTo(not_msk, CV_32F);


        _weights = (1 - alph3 / curr_K) * _weights.mul(mask) + (not_msk.mul(_weights));
        if (is_current_process_flag)
        {
            _weights = _weights + (alph3 * mask);
        }
    }

    /*
    void gaussProcess::do_classification(cv::Mat &bgmask, cv::Mat weight_mask, double weight_thresh)
    {
        cv::Mat compare_result(_img_size, _img_type);
        cv::compare(_weights, cv::Scalar(weight_thresh), compare_result, cv::CMP_GT);
        compare_result.copyTo(bgmask, weight_mask);

    }
    */
    /* END gaussProcess Functions */

    motion_segmentation::motion_segmentation(int threshold, double alpha1, double alpha2, double alpha3, double weight_thresh, uint8_t K, int init_std)
        : _thresh(threshold), _alpha1(alpha1), _alpha2(alpha2), _alpha3(alpha3), _weight_thresh(weight_thresh), _max_K(K), _init_std(init_std)
    {
        _first_frame = true;
        _curr_K = 0;
    }


    void motion_segmentation::apply(cv::InputArray image, cv::OutputArray fgmask, double) //GMM algorithm
    {
        cv::Mat mat_img = image.getMat();
        _img_type =  mat_img.type();
        _img_size = mat_img.size();

        if (_first_frame)
        {
            _first_frame = false;


            _models.insert({0, gaussProcess(mat_img, _img_type, _img_size, _init_std)});
            _curr_K++;
            _fgmask = cv::Mat::zeros(_img_size, _img_type);
            _bgmask = cv::Mat::ones(_img_size, _img_type);
            _fgmask.copyTo(fgmask);
            return;
        }

        //Обновление кадра в моделях
        for (uint8_t k = 0; k < _curr_K; k++)
        {
            _models[k].set_image(mat_img);
        }

        //Пороговая проверка пикселей на принаджежность фону
        for (uint8_t k = 0; k < _curr_K; k++)
        {
            _models[k].update_background_mask(_thresh);
        }

        //Обновление статистик моделей (для пикселей заднего фона)
        for (uint8_t k = 0; k < _curr_K; k++)
        {
            _models[k].update_statistics(_alpha1, _alpha2);
            for (uint8_t i = 0; i < _curr_K; i++)
            {
                _models[i].update_weights(_curr_K, _alpha3, _models[k].get_background_mask(), (i == k) ? true : false);
            }
        }

        _bgmask = cv::Mat::zeros(_img_size, _img_type);
        //Объединяем масски фонов
        for (uint8_t k = 0; k < _curr_K; k++)
        {
            cv::bitwise_or(_bgmask, _models[k].get_background_mask(), _bgmask);
        }

        cv::bitwise_not(_bgmask, _fgmask);

        //Если обнаружены не фоновые пиксели
        if (cv::countNonZero(_fgmask))
        {
            //Если _curr_K < _max_K То создаем новый процесс
            if (_curr_K < _max_K)
            {
                _models.insert({_curr_K, gaussProcess(mat_img, _img_type, _img_size, _init_std)});
                _curr_K++;

                //Обновляем веса, считая новый процесс текущим
                for (uint8_t i = 0; i < _curr_K; i++)
                {
                    _models[i].update_weights(_curr_K, _alpha3, _fgmask, (i == _curr_K - 1) ? true : false);
                }

            }
            //Иначе коррекция слабого процесса + классификация
            else
            {
                //Находим матрицу с минимальными весами
                cv::Mat min_weights = _models[0].get_weights();

                for (uint8_t k = 1; k < _curr_K; k++)
                {
                    cv::min(min_weights, _models[k].get_weights(), min_weights);
                }

                for (int8_t k = _curr_K - 1; k > -1; k--)
                {
                    cv::Mat weight_mask = cv::Mat(_img_size, CV_8U);
                    //Для каждого процесса создаем маску соответствия минимальных весов весам данного процесса
                    cv::compare(min_weights, _models[k].get_weights(), weight_mask, cv::CMP_EQ);
                    //Учитываем маску переднего плана (чтобы работать только с пикселями переднего плана)
                    cv::bitwise_and(weight_mask, _fgmask, weight_mask);
                    //Обновляем статистики процесса с высчитанной маской
                    _models[k].rewrite_statistics(weight_mask, _init_std);
                    //Убираем из рассмотрения в маске "min_weights" уже рассмотренные пиксели (делаем так что бы функция cv::compare для этих пикселей не выполнялась)
                    min_weights.setTo(255, weight_mask);
                }
            }
        }

        _fgmask.copyTo(fgmask);
 //std::cout << "weight_mask = " << std::endl << weight_mask << std::endl;
    }

    void motion_segmentation::setVarThreshold(int threshold)
    {
        _thresh = threshold;
    }

} // namespace cvlib
