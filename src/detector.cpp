#include <optional>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <set>
#include <iostream>
#include "../headers/detector.h"
#include "../headers/utils.h"

#define PI 3.14159265358979323846264

using namespace cv;
using namespace std;

optional<Vec2f> find_moon_point(const Mat& img) {
    int image_width = img.size().width;
    int image_height = img.size().height;

    float grid_size_ratio = 0.5;

    float stop_val = 0.05;

    while(grid_size_ratio > stop_val) {
        for(int x = image_width * grid_size_ratio; x < image_width; x+= image_width * grid_size_ratio * 2) {
            for(int y = image_height * grid_size_ratio; y < image_height; y+= image_height * grid_size_ratio * 2) {
                if(img.at<uint8_t>(round(y), round(x)) == 255) {
                    return Vec2f(x,y);
                }
            }   
        }

        grid_size_ratio/=2;
    }
    return nullopt;
}

vector<Vec2f> find_anchor_points(const Mat& img, const Vec2f& moon_point) {
    int image_width = img.size().width;
    int image_height = img.size().height;

    set<pair<float, float>> anchor_points_set;

    Vec2f border_lines[4][2] = {
        {Vec2f(0, 0), Vec2f(1, 0)}, // y = 0
        {Vec2f(0, 0), Vec2f(0, 1)}, // x = 0
        {Vec2f(0, image_height), Vec2f(1, image_height)}, // y = height
        {Vec2f(image_width, 0), Vec2f(image_width, 1)} // y = width 
    };

    float angle;
    Vec2f line_point;
    optional<Vec2f> intersect_result;
    Vec2f anchor_point;
    for(int i = 0; i < 16; i++) {
        angle = (PI/8) * i;
        line_point = Vec2f(moon_point[0] + cos(angle), moon_point[1] + sin(angle));

        for(int j = 0; j < 4; j++) {
            intersect_result = intersect(moon_point, line_point, border_lines[j][0], border_lines[j][1]);

            if(!intersect_result.has_value()) { continue; }
            anchor_point = intersect_result.value();

            if(anchor_point[0] < 0 || anchor_point[0] > image_width || anchor_point[1] < 0 || anchor_point[1] > image_height) { continue; }
            anchor_points_set.insert(pair<float, float>(anchor_point[0], anchor_point[1]));
        }
    }
    
    vector<Vec2f> anchor_points;

    auto it = anchor_points_set.begin();
    for(int i = 0; i < anchor_points_set.size(); i++) {
        anchor_points.push_back(Vec2f(it->first, it->second));
        it++;    
    }

    return anchor_points;
}

Vec2f regress_to_edge(const Mat& img, const Vec2f& point1, const Vec2f& point2) {
    Vec2f regress_point1 = Vec2f(point1[0], point1[1]);
    Vec2f regress_point2 = Vec2f(point2[0], point2[1]);

    Vec2f midpoint;
    while (distance_2d(regress_point1, regress_point2) > 1) {
        midpoint = midpoint_2d(regress_point1, regress_point2);

        // std::cout << "point1: " << regress_point1[0] << " " << regress_point1[1] << " " << unsigned(img.at<uint8_t>(round(regress_point1[1]), round(regress_point1[0]))) << "\n";
        // std::cout << "point2: " << regress_point2[0] << " " << regress_point2[1] << " " << unsigned(img.at<uint8_t>(round(regress_point2[1]), round(regress_point2[0]))) << "\n";

        if ((img.at<uint8_t>(round(midpoint[1]), round(midpoint[0])) == 255) == (img.at<uint8_t>(round(regress_point1[1]), round(regress_point1[0])) == 255)) { 
            // midpoint and rpoint1 are on same side
            regress_point1[0] = midpoint[0];
            regress_point1[1] = midpoint[1];
        } else { 
            // midpoint and rpoint2 are on the same side
            regress_point2[0] = midpoint[0];
            regress_point2[1] = midpoint[1];
        }
    }

    return midpoint_2d(regress_point1, regress_point2);
}

optional<RotatedRect> fast_contour(const Mat& img) {
    optional<Vec2f> moon_point_result = find_moon_point(img);

    if (!moon_point_result.has_value()) {
        cout << "empty\n";

        return nullopt;
    }

    Vec2f moon_point = moon_point_result.value();

    vector<Vec2f> anchor_points = find_anchor_points(img, moon_point);

    vector<Vec2f> edge_points;
    for (int i = 0; i < anchor_points.size(); i++) {
        Vec2f edge_point = regress_to_edge(img, moon_point, anchor_points.at(i));
        //cout << "Edge point " << i << ": " << edge_point[0] << " " << edge_point[1] << "\n";
        edge_points.push_back(regress_to_edge(img, moon_point, anchor_points.at(i)));
    }

    float max_aspect_ratio = -1;
    RotatedRect best_ellipse;
    float aspect_ratio;
    for(int i = 0; i < 16; i++) {
        vector<Vec2f> edge_point_subset;
        for(int j = 0; j < 6; j++) {
            edge_point_subset.push_back(edge_points.at((i+j) % 16));
        }

        RotatedRect detected_ellipse = fitEllipse(edge_point_subset);
        aspect_ratio = detected_ellipse.size.width/detected_ellipse.size.height;

        //cout << "aspect ratio: " << aspect_ratio << " bigger than max? " << max_aspect_ratio << " " << (aspect_ratio > max_aspect_ratio) << "\n";

        if(max_aspect_ratio == -1 || aspect_ratio > max_aspect_ratio) {
            max_aspect_ratio = aspect_ratio;
            best_ellipse = detected_ellipse;
        }
    }

    return best_ellipse;
}