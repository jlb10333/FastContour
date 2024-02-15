#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

void apply_brightness_contrast(         // Credit: Gavin Guo, Github: Gavin1937/MoonRegistration
    const cv::Mat& image_in,
    cv::Mat& image_out,
    int brightness,
    int contrast
)
{
    int shadow = 0;
    int highlight = 0;
    double alpha_b = 0.0;
    double gamma_b = 0.0;
    double f = 0.0;
    double alpha_c = 0.0;
    double gamma_c = 0.0;
    
    if (brightness != 0)
    {
        if (brightness > 0)
        {
            shadow = brightness;
            highlight = 255;
        }
        else
        {
            shadow = 0;
            highlight = 255 + brightness;
        }
        alpha_b = (double)(highlight - shadow)/255.0;
        gamma_b = (double)shadow;
        
        cv::addWeighted(image_in, alpha_b, image_in, 0, gamma_b, image_out);
    }
    else
        image_in.copyTo(image_out);
    
    if (contrast != 0)
    {
        f = 131.0*((double)contrast + 127.0)/(127.0*(131.0-(double)contrast));
        alpha_c = f;
        gamma_c = 127.0*(1.0-f);
        
        cv::addWeighted(image_out, alpha_c, image_out, 0, gamma_c, image_out);
    }
}

void default_preprocess_steps(          // Credit: Gavin Guo, Github: Gavin1937/MoonRegistration
    const cv::Mat& image_in,
    cv::Mat& image_out
)
{
    
    // creating gray scale version of image needed for HoughCircles
    cv::cvtColor(image_in, image_out, cv::COLOR_BGR2GRAY);
    
    // set img to maximum contrast
    // only leave black & white pixels
    // usually, moon will be white after this conversion
    cv::Mat buff;
    apply_brightness_contrast(image_out, buff, 0, 30);
    
    // set gray image to black & white only image by setting its threshold
    // opencv impl of threshold
    // dst[j] = src[j] > thresh ? maxval : 0;
    // using threshold to binarize img will rm all the branching when calculating img_brightness_perc
    // which make calculation much faster
    cv::threshold(buff, image_out, 0, 255, cv::THRESH_BINARY);
}