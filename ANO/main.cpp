#include "stdafx.h"
#include <math.h> 
#include <string> 
#include <opencv2/opencv.hpp>
#include "ObjectData.h"


#define M_PI 3.14159265358979323846
using namespace std;

double ComputePerimeterForObject(FeatureList &obj, cv::Mat indexedImage, cv::Mat coloredImage, int p, int q)
{
	double perimeter = 0.0;
	int index = obj.Index;
	cv::Point center = obj.Center;
	for (int y = 0; y < indexedImage.rows; y++) {
		for (int x = 0; x < indexedImage.cols; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				perimeter += pow((x - center.x), p) * pow((y - center.y), q) * index;
			}
		}
	}
	return perimeter;
}

void ComputePerimeter(ObjectData &feature)
{
	std::cout << "Computing perimeters" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		(*obj).Perimeter = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 0);
		obj++;
	}

	std::cout << "Done ..." << std::endl;

}

void ComputeFeatureOne(ObjectData &feature)
{
	std::cout << "Computing feature 1" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		(*obj).Feature1 = pow((*obj).Perimeter, 2) / (100 * (*obj).Area);
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}

void ComputeFeatureTwo(ObjectData &feature)
{
	std::cout << "Computing feature 2" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		double micro20 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 2, 0);
		double micro02 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 2);
		double micro11 = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 1, 1);
		double microMax = (1 / 2) * (micro20 + micro02) + (1 / 2) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));
		double microMin = (1 / 2) * (micro20 + micro02) - (1 / 2) * sqrt((4 * pow(micro11, 2)) + pow(micro20 - micro02, 2));

		(*obj).Feature2 = microMin / microMax;
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}

void Image_index(cv::Mat &image, cv::Mat &indexedImage, cv::Mat &coloredImage, int y, int x, int index, cv::Vec3b color) {
	if (x > image.cols || x < 0)
		return;
	if (y > image.rows || y < 0)
		return;

	if (image.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
	{
		indexedImage.at<float>(y, x) = index;
		coloredImage.at<cv::Vec3b>(y, x) = color;
		Image_index(image, indexedImage, coloredImage, y, x + 1, index, color);
		Image_index(image, indexedImage, coloredImage, y + 1, x, index, color);
		Image_index(image, indexedImage, coloredImage, y, x - 1, index, color);
		Image_index(image, indexedImage, coloredImage, y - 1, x, index, color);
	}
}

void Calc_center(cv::Mat &indexedImage, cv::Mat &coloredImage, std::list<int> indexes) {
	if (indexes.size() > 0)
	{
		int index = 0;
		std::list<int>::iterator it = indexes.begin();
		std::list<cv::Point> centers;

		while (it != indexes.end())
		{
			index = *it;
			float m10 = 0.0;
			float m01 = 0.0;
			float m00 = 0.0;
			for (int y = 0; y < indexedImage.rows; y++) {
				for (int x = 0; x < indexedImage.cols; x++) {
					if (indexedImage.at<float>(y, x) == index) {
						m10 += pow(x, 1) * pow(y, 0) * indexedImage.at<float>(y, x);
						m01 += pow(x, 0) * pow(y, 1) * indexedImage.at<float>(y, x);
						m00 += pow(x, 0) * pow(y, 0) * indexedImage.at<float>(y, x);
					}
				}
			}
			centers.push_back(cv::Point(m10 / m00, m01 / m00));

			it++;
		}
		std::list<cv::Point>::iterator centerIterator = centers.begin();
		int counter = 0;
		for (centerIterator = centers.begin(); centerIterator != centers.end(); centerIterator++)
		{
			int x = centerIterator->x;
			int y = centerIterator->y;
			//string centerPoint = to_string(x) + ":" + to_string(y);
			string text = std::to_string(counter++);
			cv::putText(coloredImage,
				text,
				cv::Point(x, y), // Coordinates
				cv::FONT_ITALIC, // Font
				0.5, // Scale. 2.0 = 2x bigger
				cv::Scalar(255, 255, 255)); // BGR Color

		}
	}
}

void ComputeCenterOfObjects(ObjectData &feature) {

	int index = 0;
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	std::list<FeatureList> reducedObjects;
	std::cout << "Computing centers" << std::endl;
	while (obj != feature.Objects.end())
	{
		float m10 = 0.0;
		float m01 = 0.0;
		float m00 = 0.0;
		for (int y = 0; y < feature.IndexedImage.rows; y++) {
			for (int x = 0; x < feature.IndexedImage.cols; x++) {
				if (feature.IndexedImage.at<float>(y, x) == (*obj).Index) {
					m10 += pow(x, 1) * pow(y, 0) * feature.IndexedImage.at<float>(y, x);
					m01 += pow(x, 0) * pow(y, 1) * feature.IndexedImage.at<float>(y, x);
					m00 += pow(x, 0) * pow(y, 0) * feature.IndexedImage.at<float>(y, x);
				}
			}
		}
		if (m00 > 100)
		{
			cv::Point p = cv::Point(m10 / m00, m01 / m00);
			(*obj).Center = p;
			(*obj).Area = m00;
			reducedObjects.push_back(*obj);
		}

		obj++;
	}
	feature.Objects = reducedObjects;
	std::cout << "Done ... " << std::endl;

}

void ImageIndexing()
{
	cv::Mat image = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat imageGray;
	image.convertTo(imageGray, CV_32FC1, 1.0 / 255.0);
	cv::Mat indexedImage = cv::Mat::zeros(image.size(), CV_32FC1);
	cv::Mat coloredImage = cv::Mat::zeros(image.size(), CV_8UC3);
	std::list<int> indexes;

	std::list<FeatureList> objects;

	int counter = 0;
	for (int y = 0; y < imageGray.rows; y++) {
		for (int x = 0; x < imageGray.cols; x++) {
			if (imageGray.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				cv::Vec3b color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
				indexes.push_back(counter);
				FeatureList obj = FeatureList(counter, color);
				objects.push_back(obj);
				Image_index(imageGray, indexedImage, coloredImage, y, x, counter, color);
				counter++;
			}
		}
	}


	ObjectData feature;
	feature.IndexedImage = indexedImage;
	feature.ColoredImage = coloredImage;
	feature.Objects = objects;

	ComputeCenterOfObjects(feature);
	ComputePerimeter(feature);
	ComputeFeatureOne(feature);
	ComputeFeatureTwo(feature);

	//Calc_center(indexedImage, coloredImage, indexes);

	cv::imshow("Original", imageGray);
	cv::imshow("Threshold", feature.IndexedImage);
	cv::imshow("Colored", feature.ColoredImage);


	cv::waitKey(0);


}



int main()
{
	//Excercise 1
	ImageIndexing();
}