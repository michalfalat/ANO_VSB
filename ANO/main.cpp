#include "stdafx.h"
#include <math.h> 
#include <string> 
#include <opencv2/opencv.hpp>
#include "ObjectData.h"


#define M_PI 3.14159265358979323846
using namespace std;


double CalculateMoment(FeatureList &obj, cv::Mat indexedImage, cv::Mat coloredImage, int p, int q)
{
	double moment = 0.0;
	int index = obj.Index;
	cv::Point center = obj.Center;
	for (int y = 0; y < indexedImage.rows; y++) {
		for (int x = 0; x < indexedImage.cols; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				moment += pow(x, p) * pow(y, q);
			}
		}
	}
	return moment;
}

double ComputeAreaForObject(FeatureList &obj, cv::Mat indexedImage, cv::Mat coloredImage)
{
	double area = 0.0;
	int index = obj.Index;
	cv::Point center = obj.Center;
	for (int y = 0; y < indexedImage.rows; y++) {
		for (int x = 0; x < indexedImage.cols; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				area += pow(x, 0) * pow(y, 0);
			}
		}
	}
	return area;
}

double ComputePerimeterForObject(FeatureList &obj, cv::Mat indexedImage, cv::Mat coloredImage, int p, int q)
{
	double perimeter = 0.0;
	int index = obj.Index;
	cv::Point center = obj.Center;
	for (int y = 1; y < indexedImage.rows - 1; y++) {
		for (int x = 1; x < indexedImage.cols - 1; x++) {


			if (indexedImage.at<float>(y, x) == index) {
				if(indexedImage.at<float>(y, x + 1) == index &&
					indexedImage.at<float>(y, x - 1) == index &&
					indexedImage.at<float>(y + 1, x) == index &&
					indexedImage.at<float>(y - 1, x) == index) {
				}
				else {
					perimeter += 1;
				}
			}
				
		}
	}
	return perimeter;
}

void ComputePerimeterAndArea(ObjectData &feature)
{
	std::cout << "Computing perimeters" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		double per = ComputePerimeterForObject((*obj), feature.IndexedImage, feature.ColoredImage, 0, 0);
		double area = ComputeAreaForObject((*obj), feature.IndexedImage, feature.ColoredImage);
		(*obj).Perimeter = per;
		(*obj).Area = area;
		std::cout << "Object " << (*obj).Index  << " perimeter: " << per  << " area: " << area << std::endl;
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

double ComputeEuclideanDistance(MyPoint a, Ethalon b)
{
	double distance = sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
	return distance;
}

double ComputeEuclideanDistance(MyPoint a, MyPoint b)
{
	double distance = sqrt(pow(b.x - a.x, 2) + pow(b.y - a.y, 2));
	return distance;
}

void ComputeEthalons(ObjectData &feature)
{
	std::cout << "Computing  ethalons" << std::endl;

	double x = 0.0;
	double y = 0.0;
	std::list<Ethalon> ethalons;
	Ethalon currentEthalon = Ethalon(0.0, 0.0);
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		//feature1 => x
		//feature2 => y

		if (currentEthalon.x == 0.0)
		{
			currentEthalon = Ethalon((*obj).Feature1, (*obj).Feature2);
			ethalons.push_back(currentEthalon);
		}
		else
		{
			MyPoint currentPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
			std::list<Ethalon>::iterator eth = ethalons.begin();
			bool found = false;
			while (eth != ethalons.end())
			{
				if (ComputeEuclideanDistance(currentPoint, (*eth)) < 0.2)
				{
					(*eth) = Ethalon((currentPoint.x + (*eth).x) / 2, (currentPoint.y + (*eth).y) / 2);
					found = true;
				}

				eth++;
			}
			if (!found)
			{
				ethalons.push_back(Ethalon(currentPoint.x, currentPoint.y));
			}
		}
		obj++;
	}
	std::list<Ethalon>::iterator eth = ethalons.begin();
	while (eth != ethalons.end())
	{
		(*eth).AddClass();
		eth++;
	}

	feature.Ethalons = ethalons;
	std::cout << "Done ..." << std::endl;

}

void AssignClassToObject(ObjectData &feature)
{
	std::cout << "Getting class for objects" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		MyPoint objectPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
		double mindst = INFINITY;
		Ethalon closestEthalon;
		std::list<Ethalon>::iterator eth = feature.Ethalons.begin();
		while (eth != feature.Ethalons.end())
		{
			double dst = ComputeEuclideanDistance(objectPoint, (*eth));
			if (dst < mindst)
			{
				mindst = dst;
				closestEthalon = (*eth);
			}
			eth++;
		}
		(*obj).ClassLabel = closestEthalon;
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}



void ComputeKMeans(ObjectData &feature)
{
	std::cout << "Computing k-means" << std::endl;

	int numOfCentroids = feature.Ethalons.size();
	srand(time(NULL));
	std::list<CentroidObject> centroids;

	for (int i = 0; i < numOfCentroids; i++)
	{
		int index = rand() % (feature.Objects.size() - 1) + 1;
		std::list<FeatureList>::iterator it = feature.Objects.begin();
		std::advance(it, index);
		CentroidObject c;
		c.Centroid = MyPoint((*it).Feature1, (*it).Feature2);
		centroids.push_back(c);
	}
	double delta = 0.05;
	bool iterate = true;

	while (iterate)
	{
		//clear closestobject list from previous iteration
		std::list<CentroidObject>::iterator cen = centroids.begin();
		while (cen != centroids.end())
		{
			(*cen).ClosestObjects.clear();
			cen++;
		}

		//asign objects to centroids
		std::list<FeatureList>::iterator obj = feature.Objects.begin();
		while (obj != feature.Objects.end())
		{
			double distance = INFINITY;
			CentroidObject *closestCentroid = nullptr;
			MyPoint objectPoint = MyPoint((*obj).Feature1, (*obj).Feature2);
			std::list<CentroidObject>::iterator cen = centroids.begin();
			while (cen != centroids.end())
			{
				double temp = ComputeEuclideanDistance((*cen).Centroid, objectPoint);
				if (temp < distance)
				{
					distance = temp;
					closestCentroid = &(*cen);
				}

				cen++;
			}
			closestCentroid->ClosestObjects.push_back((*obj));
			obj++;
		}

		int tmp = 0;
		//compute centroids
		cen = centroids.begin();
		while (cen != centroids.end())
		{
			if ((*cen).ClosestObjects.size() > 0)
			{
				MyPoint oldCentroid = (*cen).Centroid;
				double sumX = 0.0;
				double sumY = 0.0;
				int count = 0;
				std::list<FeatureList>::iterator obj = (*cen).ClosestObjects.begin();
				while (obj != (*cen).ClosestObjects.end())
				{
					sumX += (*obj).Feature1;
					sumY += (*obj).Feature2;
					count++;
					obj++;
				}

				MyPoint newCentroid = MyPoint(sumX / count, sumY / count);
				double dist = ComputeEuclideanDistance(oldCentroid, newCentroid);
				if (dist <= delta)
				{
					if (tmp == 0)
						iterate = false;
				}
				else
				{
					iterate = true;
				}

				(*cen).Centroid = newCentroid;

				tmp++;
			}

			cen++;
		}
	}

	feature.Centroids = centroids;
	std::cout << "Done ..." << std::endl;

}	



void ComputeFeatureTwo(ObjectData &feature)
{
	std::cout << "Computing feature 2" << std::endl;

	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		double m20 = CalculateMoment((*obj), feature.IndexedImage, feature.ColoredImage, 2, 0);
		double m02 = CalculateMoment((*obj), feature.IndexedImage, feature.ColoredImage, 0, 2);
		double m11 = CalculateMoment((*obj), feature.IndexedImage, feature.ColoredImage, 1, 1);
		double microMax = 0.5f * (m20 + m02) +  (0.5f * sqrt((4 * pow(m11, 2)) + pow(m20 - m02, 2)));
		double microMin = 0.5f * (m20 + m02) -  (0.5f * sqrt((4 * pow(m11, 2)) + pow(m20 - m02, 2)));

		(*obj).Feature2 = microMin / microMax;
		obj++;
	}
	std::cout << "Done ..." << std::endl;

}



void ShowFeatures(ObjectData &feature)
{
	cv::Mat coloredImage = cv::Mat::zeros(cv::Size(600, 200), CV_8UC3);
	std::list<FeatureList>::iterator obj = feature.Objects.begin();
	while (obj != feature.Objects.end())
	{
		circle(coloredImage, cv::Point((*obj).Feature1 * 500, (*obj).Feature2 * 50), 3, cv::Scalar(0.0, 0.0, 255.0, 1.0), 1);
		obj++;
	}

	std::list<CentroidObject>::iterator cen = feature.Centroids.begin();
	while (cen != feature.Centroids.end())
	{
		circle(coloredImage, cv::Point((*cen).Centroid.x * 500, (*cen).Centroid.y * 50), 6, cv::Scalar(255.0, 0.0, 0.0, 1.0), 1);
		cen++;
	}

	imshow("Features", coloredImage);


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
	ComputePerimeterAndArea(feature);
	ComputeFeatureOne(feature);
	ComputeFeatureTwo(feature);

	ComputeEthalons(feature);
	AssignClassToObject(feature);
	ComputeKMeans(feature);

	cv::imshow("Original", imageGray);
	cv::imshow("Threshold", feature.IndexedImage);
	cv::imshow("Colored", feature.ColoredImage);

	ShowFeatures(feature);


	cv::waitKey(0);


}



int main()
{
	//Excercise 1
	ImageIndexing();
}