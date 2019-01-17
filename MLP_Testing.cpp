#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <fstream>

using namespace cv;
using namespace ml;
using namespace std;

Ptr<ANN_MLP> mlp;
Mat test_sample;
int correct_class = 0;
int wrong_class = 0;
int false_positives[10] = { 0,0,0,0,0,0,0,0,0,0 };
Point max_loc = Point(0, 0);


void Neuralnetwrok(Mat inputTrainingData, Mat outputTrainingData)
{


		double start = (double)cv::getTickCount();
		cout << "Training" << endl;

		Ptr<ANN_MLP> mlp = ANN_MLP::create();
		const int hiddenLayerSize = 100;

		Mat layersSize = Mat(3, 1, CV_16U);
		layersSize.row(0) = Scalar(inputTrainingData.cols);
		layersSize.row(1) = Scalar(hiddenLayerSize);
		layersSize.row(2) = Scalar(outputTrainingData.cols);
		mlp->setLayerSizes(layersSize);
		mlp->getLayerSizes();
		mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM,0, 1);

		TermCriteria termCrit = TermCriteria(
			TermCriteria::Type::COUNT + TermCriteria::Type::EPS,
			1000,
			0.000001
		);
		mlp->setTermCriteria(termCrit);


	



		mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
		Ptr<TrainData> trainingData = TrainData::create(
			inputTrainingData,
			SampleTypes::ROW_SAMPLE,
			outputTrainingData*1.5 
		);

		mlp->train(trainingData
			/*, ANN_MLP::TrainFlags::UPDATE_WEIGHTS
			+ ANN_MLP::TrainFlags::NO_INPUT_SCALE
			+ ANN_MLP::TrainFlags::NO_OUTPUT_SCALE*/
		);


		cout << "saving" << endl;
		mlp->save("Add__feature__leaf_dataset_balanced_premeter.xml");



		// Test row data 
		int row_feature = inputTrainingData.rows;

		for (int i = 0; i < row_feature; i++) {

			Mat sample = inputTrainingData.row(i);
			Mat Classs = outputTrainingData.row(i);

			Mat result;
			mlp->predict(sample, result);

			cout << result;

			minMaxLoc(result, 0, 0, 0, &max_loc);


			cout << "i:" << i << sample << Classs << " ----> " << max_loc.x;
			cout << endl;

			ofstream outputFile;
			outputFile.open("outputresult.csv", ios_base::app);
			outputFile << format(max_loc.x, cv::Formatter::FMT_CSV) << "\n";
			outputFile.close();

	


			if (!(outputTrainingData.at<float>(i, max_loc.x)))
			{
				// if they differ more than floating point error => wrong class

				wrong_class++;

				false_positives[(int)max_loc.x]++;

			}
			else
			{

				// otherwise correct

				correct_class++;
				//cout << i;
			}
		}
		
		
		cout << correct_class << endl;
		cout << wrong_class << endl;
		cout <<"Correct Class: " << (double)correct_class * 100 / 887 << endl;
		cout <<"Wrong   Class: " << (double)wrong_class * 100 / 887 << endl;

		for (int i = 0; i < 10; i++)
		{

			cout << i << false_positives[i] << "-->" << (double)false_positives[i] * 100 / 887;
		}
		// Time Spent

		std::cout << "It took: " << ((double)cv::getTickCount() - start) / cv::getTickFrequency() / 60 << " minutes."
			<< std::endl;

	
	
}

void test(Mat test_data)
{
	mlp = StatModel::load<ANN_MLP>("leaf.xml");
	
	int row_feature = test_data.rows;

	for (int i = 0; i < row_feature; i++) {
		Mat results;
		Mat sample = test_data.row(i);
		mlp->predict(sample, results);
		minMaxLoc(results, 0, 0, 0, &max_loc);

		cout << "i:" << i << sample << " ----> " << max_loc.x;
		cout << endl;
		
	}
}

int main()
{

	cv::Ptr<cv::ml::TrainData> input = cv::ml::TrainData::loadFromCSV("dataSet.csv", 0, -2, 0);
	cv::Mat data = input->getSamples();

	Mat Class,label;
	data.copyTo(Class);
	Mat Class_label = Class.colRange(0,9);
	int row_class = Class_label.rows;
	int col_class = Class_label.cols;
	Mat outputTrainingData = (row_class, col_class, Class_label);
	
	


	Mat feature_no_class;
	data.copyTo(feature_no_class);
	Mat feature = feature_no_class.colRange(9, 15);
	int row_feature = feature.rows;
	int col_feature = feature.cols;
	

	Mat inputTrainingData = (row_feature, col_feature, feature);
	//Mat inputTestData = (row_feature, col_feature, feature);

	//cout << inputTrainingData;

	//Neuralnetwrok(inputTrainingData, outputTrainingData);

	int choice;
	do {
		cout << "";
		cin >> choice;

		switch (choice)
		{
		case 1: test(inputTrainingData);
			break;
		case 2: Neuralnetwrok(inputTrainingData, outputTrainingData);
			break;

		}
	} while (choice != 0);


	system("PAUSE");


}
