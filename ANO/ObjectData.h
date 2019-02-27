#pragma once

#include <opencv2/opencv.hpp>

class FeatureList {
public:
	double Index;
	double Area;
	cv::Vec3b Color;
	cv::Point Center;
	double Perimeter;
	double Feature1;
	double Feature2;

	FeatureList(int index, cv::Vec3b color) {
		this->Index = index;
		this->Color = color;
	}

	static double GetPerimeter(FeatureList obj) { return obj.Perimeter; }
};

class ObjectData {
public:
	cv::Mat IndexedImage;
	cv::Mat ColoredImage;
	std::list<FeatureList> Objects;


};