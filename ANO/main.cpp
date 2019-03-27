#include "stdafx.h"
#include <math.h> 
#include <string> 
#include <opencv2/opencv.hpp>
#include "ObjectData.h"


#define LAMBDA 1.0
#define ETA 0.1

#define SQR( x ) ( ( x ) * ( x ) )
#define M_PI 3.14159265358979323846
using namespace std;


void randomize(double * p, int n)
{
	for (int i = 0; i < n; i++) {
		p[i] = (double)rand() / (RAND_MAX);
	}
}

NN * createNN(int n, int h, int o)
{
	srand(time(NULL));
	NN * nn = new NN;

	nn->n = new int[3];
	nn->n[0] = n;
	nn->n[1] = h;
	nn->n[2] = o;
	nn->l = 3;

	nn->w = new double **[nn->l - 1];


	for (int k = 0; k < nn->l - 1; k++)
	{
		nn->w[k] = new double *[nn->n[k + 1]];
		for (int j = 0; j < nn->n[k + 1]; j++)
		{
			nn->w[k][j] = new double[nn->n[k]];
			randomize(nn->w[k][j], nn->n[k]);
			// BIAS
			//nn->w[k][j] = new double[nn->n[k] + 1];			
			//randomize( nn->w[k][j], nn->n[k] + 1 );
		}
	}

	nn->y = new double *[nn->l];
	for (int k = 0; k < nn->l; k++) {
		nn->y[k] = new double[nn->n[k]];
		memset(nn->y[k], 0, sizeof(double) * nn->n[k]);
	}

	nn->in = nn->y[0];
	nn->out = nn->y[nn->l - 1];

	nn->d = new double *[nn->l];
	for (int k = 0; k < nn->l; k++) {
		nn->d[k] = new double[nn->n[k]];
		memset(nn->d[k], 0, sizeof(double) * nn->n[k]);
	}

	return nn;
}

void releaseNN(NN *& nn)
{
	for (int k = 0; k < nn->l - 1; k++) {
		for (int j = 0; j < nn->n[k + 1]; j++) {
			delete[] nn->w[k][j];
		}
		delete[] nn->w[k];
	}
	delete[] nn->w;

	for (int k = 0; k < nn->l; k++) {
		delete[] nn->y[k];
	}
	delete[] nn->y;

	for (int k = 0; k < nn->l; k++) {
		delete[] nn->d[k];

	}
	delete[] nn->d;

	delete[] nn->n;

	delete nn;
	nn = NULL;
}

void feedforward(NN * nn)
{
	for (int k = 0; k < nn->l; k++)
	{
		int neuronsCount = nn->n[k];
		if (k == 0)
		{
			for (int n = 0; n < neuronsCount; n++)
			{
				nn->y[k][n] = 1.0 / (1.0 + exp(-LAMBDA * nn->in[n]));
			}
		}
		else
		{
			for (int n = 0; n < neuronsCount; n++)
			{
				double weight = 0.0;
				for (int we = 0; we < nn->n[k - 1]; we++)
				{
					weight += nn->w[k - 1][n][we] * nn->y[k - 1][we];
				}

				weight = 1.0 / (1.0 + exp(-LAMBDA * weight));
				nn->y[k][n] = weight;
			}
		}
	}
	for (int o = 0; o < nn->n[nn->l - 1]; o++)
	{
		nn->out[o] = nn->y[nn->l - 1][o];
	}



}

double backpropagation(NN * nn, double * t)
{

	double error = 0.0;

	for (int k = nn->l - 1; k >= 0; k--)
	{
		if (k == nn->l - 1)
		{
			for (int n = 0; n < nn->n[k]; n++)
			{
				double to = t[n];
				double y = nn->y[k][n];
				double res = (to - nn->y[k][n]) * LAMBDA *   nn->y[k][n] * (1 - nn->y[k][n]);
				nn->d[k][n] -= res;
			}
		}
		else
		{
			for (int n = 0; n < nn->n[k]; n++)
			{
				double errorResult = 0.0;
				int indexOfUpperlayer = k + 1;
				for (int j = 0; j < nn->n[indexOfUpperlayer]; j++)
				{
					errorResult += nn->d[indexOfUpperlayer][j] * nn->w[k][j][n];
				}

				errorResult = errorResult * LAMBDA *   nn->y[k][n] * (1 - nn->y[k][n]);
				nn->d[k][n] -= errorResult;
			}
		}
	}

	for (int k = 0; k < nn->l - 1; k++)//layers
	{
		for (int i = 0; i < nn->n[k + 1]; i++)//upper layer
		{
			for (int j = 0; j < nn->n[k]; j++)//lower layer
			{
				double oldWeight = nn->w[k][i][j];
				double d = nn->d[k + 1][i];
				double y = nn->y[k][i];
				double newWeight = ETA * d * y;
				nn->w[k][i][j] += newWeight;
			}
		}
	}


	for (int n = 0; n < nn->n[nn->l - 1]; n++)
	{
		error += pow(t[n] - nn->y[nn->l - 1][n], 2);
	}
	error = error * (1.0 / 2.0);

	return error;
}

void setInput(NN * nn, double * in, bool verbose)
{
	memcpy(nn->in, in, sizeof(double) * nn->n[0]);

	if (verbose) {
		printf("input=(");
		for (int i = 0; i < nn->n[0]; i++) {
			printf("%0.3f", nn->in[i]);
			if (i < nn->n[0] - 1) {
				printf(", ");
			}
		}
		printf(")\n");
	}
}

int getOutput(NN * nn, bool verbose)
{
	double max = 0.0;
	int max_i = 0;
	if (verbose) printf(" output=");
	for (int i = 0; i < nn->n[nn->l - 1]; i++)
	{
		if (verbose) printf("%0.3f ", nn->out[i]);
		if (nn->out[i] > max) {
			max = nn->out[i];
			max_i = i;
		}
	}
	if (verbose) printf(" -> %d\n", max_i);
	if (nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1) return 2;
	return max_i;
}



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


void train(NN* nn)
{
	int n = 1000;
	double ** trainingSet = new double *[n];
	for (int i = 0; i < n; i++) {
		trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];

		bool classA = i % 2;

		for (int j = 0; j < nn->n[0]; j++) {
			if (classA) {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else {
				trainingSet[i][j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}

		trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
		trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
	}

	double error = 1.0;
	int i = 0;
	while (error > 0.001)
	{
		setInput(nn, trainingSet[i%n], true);
		feedforward(nn);
		error = backpropagation(nn, &trainingSet[i%n][nn->n[0]]);
		i++;
		printf("\rerr=%0.3f", error);
	}
	printf(" (%d iterations)\n", i);

	for (int i = 0; i < n; i++) {
		delete[] trainingSet[i];
	}
	delete[] trainingSet;
}

void test(NN* nn, int num_samples = 10)
{
	double* in = new double[nn->n[0]];

	int num_err = 0;
	for (int n = 0; n < num_samples; n++)
	{
		bool classA = rand() % 2;

		for (int j = 0; j < nn->n[0]; j++)
		{
			if (classA)
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.6;
			}
			else
			{
				in[j] = 0.1 * (double)rand() / (RAND_MAX)+0.2;
			}
		}
		printf("predicted: %d\n", !classA);
		setInput(nn, in, true);

		feedforward(nn);
		int output = getOutput(nn, true);
		if (output == classA) num_err++;
		printf("\n");
	}
	double err = (double)num_err / num_samples;
	printf("test error: %.2f\n", err);
}


int main()
{
	//Excercise 1- 6
	// ImageIndexing();
	NN * nn = createNN(2, 4, 2);
	    train(nn);
	    
	    getchar();
	    
	    test(nn, 100);
	
		getchar();
	
		releaseNN( nn );

		cv::waitKey(0);
		return 0;

}