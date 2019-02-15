#include "stdafx.h"
#include <math.h> 
#include <string> 
#include <opencv2/opencv.hpp>


#define M_PI 3.14159265358979323846
using namespace std;

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

void ImageIndexing()
{
	cv::Mat image = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat imageGray;
	image.convertTo(imageGray, CV_32FC1, 1.0 / 255.0);
	cv::Mat indexedImage = cv::Mat::zeros(image.size(), CV_32FC1);
	cv::Mat coloredImage = cv::Mat::zeros(image.size(), CV_8UC3);
	std::list<int> indexes;
	int counter = 0;
	for (int y = 0; y < imageGray.rows; y++) {
		for (int x = 0; x < imageGray.cols; x++) {
			if (imageGray.at<float>(y, x) != 0 && indexedImage.at<float>(y, x) == 0)
			{
				cv::Vec3b color = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
				indexes.push_back(counter);
				Image_index(imageGray, indexedImage, coloredImage, y, x, counter, color);
				counter++;
			}
		}
	}

	Calc_center(indexedImage, coloredImage, indexes);

	cv::imshow("Original", imageGray);
	cv::imshow("Threshold", indexedImage);
	cv::imshow("Colored", coloredImage);


	cv::waitKey(0);


}



int main()
{
	//Excercise 1
	ImageIndexing();
}