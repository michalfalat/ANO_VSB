#pragma once

#include <opencv2/opencv.hpp>


class Ethalon {
public:
	double x;
	double y;
	std::string label;
	Ethalon(double x, double y)
	{
		this->x = x;
		this->y = y;
	}

	Ethalon() {}

	void AddClass() {

		if (this->x >= 0.11 && this->x <= 0.18 && this->y >= 0.85 && this->y <= 0.99)
		{
			this->label = "Square";
		}
		else if (this->x >= 0.14  && this->x <= 0.21 && this->y >= 0.02 && this->y <= 0.25)
		{
			this->label = "Rectangle";
		}
		else if (this->x >= 0.58 && this->x <= 0.70 && this->y >= 0.7  && this->y <= 0.99)
		{
			this->label = "Star";
		}
		else
		{
			this->label = "Unknown";
		}

	}
};

class FeatureList {
public:
	double Index;
	double Area;
	cv::Vec3b Color;
	cv::Point Center;
	double Perimeter;
	double Feature1;
	double Feature2;
	Ethalon ClassLabel;

	FeatureList(int index, cv::Vec3b color) {
		this->Index = index;
		this->Color = color;
	}

	static double GetPerimeter(FeatureList obj) { return obj.Perimeter; }
};


struct NN {
	int * n; // pocty neuronu
	int l; // pocet vrstev
	double *** w; // vahy

	double * in; // vstupni vektor
	double * out; // vystupni vektor
	double ** y; // vystupni vektory vrstev

	double ** d; // chyby neuronu
};
//
//NN * createNN(int n, int h, int o);
//void releaseNN(NN *& nn);
//void feedforward(NN * nn);
//double backpropagation(NN * nn, double * t);
//void setInput(NN * nn, double * in, bool verbose = false);
//int getOutput(NN * nn, bool verbose = false);


class MyPoint {
public:
	double x;
	double y;
	MyPoint() {}
	MyPoint(double x, double y)
	{
		this->x = x;
		this->y = y;
	}
};




class CentroidObject {
public:
	MyPoint Centroid;
	std::list<FeatureList> ClosestObjects;
	//}
	CentroidObject() {}
};


class ObjectData {
public:
	cv::Mat IndexedImage;
	cv::Mat ColoredImage;
	std::list<FeatureList> Objects;

	std::list<Ethalon> Ethalons;
	std::list<CentroidObject> Centroids;


};