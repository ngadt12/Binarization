#include "stdafx.h"

#include <stdio.h>	// For printf()
#include <opencv\cv.h>		// Main OpenCV library.
#include <opencv\highgui.h>	// OpenCV functions for files and graphical windows.
#include "opencv2/imgproc/imgproc.hpp"
#include "BinCalc.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>       // std::vector



using namespace cv; // cv::
using namespace std; // std::

/**
* @function main
*/


int main(int argc, const char** argv)
{
	char filename[256];
	char outputName[256];

	int nWidth, nHeight; 
	//sprintf(filename, "./input/1_1_.png");
	//sprintf(filename, "./input/2_1_.png");
	//sprintf(filename, "./input/4_1_.png");
	//sprintf(filename, "./input/1280x960org.png"); 
	//sprintf(filename, "./input/1280x960org_.png");
	//sprintf(filename, "./input/1original.png");
	//sprintf(filename, "./input/SauvolaCode.png");
	//sprintf(filename, "./input/H04.png");
	//sprintf(filename, "./input/H02_.png");
	//sprintf(filename, "./input/6_1_.png");
	//sprintf(filename, "./input/3_1__.png");
	//sprintf(filename, "./input/H-DIBCO2012-dataset/H13_.png");
	//sprintf(filename, "./input/DIBCO2013_Dataset/PR4_.png");
	//sprintf(filename, "./input/1280x960org-1.png");
	//sprintf(filename, "./input/H05_.png");
	//sprintf(filename, "./input/2Capture__.png");
	//sprintf(filename, "./input/H11__.png");  //noise
	//sprintf(filename, "./input/5_1_.png");
	//sprintf(filename, "./input/H13_.png");  //noise
	//sprintf(filename, "./input/20150928_124411.png");  //noise
	//sprintf(filename, "./input/test3_.png");   sprintf(outputName, "./output/test3_.png");
	//sprintf(filename, "./input/test2.png");    sprintf(outputName, "./output/test2.png");
	//sprintf(filename, "./input/2_1.png");  //noise  sprintf(outputName, "./input/HDICO2014/original_images/oH01.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/Challenge1.png");  sprintf(outputName, "./output/Challenge1.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/HDICO2014/original_images/H02.png");  sprintf(outputName, "./input/HDICO2014/original_images/oH02.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/DIBCO2013_Dataset/OriginalImages/HW5.bmp");  sprintf(outputName, "./input/DIBCO2013_Dataset/OriginalImages/oHW5.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/DIBCO2013_Dataset/OriginalImages/PR3.bmp");  sprintf(outputName, "./input/DIBCO2013_Dataset/OriginalImages/oPR3.png");  //noise= "./output/test2.png";
	
	//sprintf(filename, "./input/IDCAR2015/img_1.jpg");  sprintf(outputName, "./input/IDCAR2015/oimg_1.jpg");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/IDCAR2015/img_40.jpg");  sprintf(outputName, "./input/IDCAR2015/oimg_40.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/IDCAR2015/img_33.png");  sprintf(outputName, "./input/IDCAR2015/oimg_33.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/DBICO2011/DIBCO11-handwritten/HW4.png");  sprintf(outputName, "./input/DBICO2011/DIBCO11-handwritten/oHW4.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/DBICO2009/DIBC02009_Test_images-handwritten/H05.bmp");  sprintf(outputName, "./input/DBICO2009/DIBC02009_Test_images-handwritten/oH05.bmp");  //noise= "./output/test2.png";
	sprintf(filename, "./input/DBICO2009/DIBCO2009_Test_images-printed/P01.bmp");  sprintf(outputName, "./input/DBICO2009/DIBCO2009_Test_images-printed/oP01.bmp");  //noise= "./output/test2.png";
	
	//sprintf(filename, "./input/temp/11billionpeople.jpg");  sprintf(outputName, "./input/temp/o11billionpeople.jpg");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/1original.jpg");  sprintf(outputName, "./input/temp/o1original.jpg");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/2Capture_.PNG");  sprintf(outputName, "./input/temp/o2Capture_.PNG");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/H05.bmp");  sprintf(outputName, "./input/temp/oH05.bmp");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/test3_.png");  sprintf(outputName, "./input/temp/otest3_.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/H04.bmp");  sprintf(outputName, "./input/temp/oH04.bmp");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/H11__.png");  sprintf(outputName, "./input/temp/oH11__.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/1_1.png");  sprintf(outputName, "./input/temp/o1_1.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/Capture1.png");  sprintf(outputName, "./input/temp/oCapture1.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/f1.png");  sprintf(outputName, "./input/temp/of1.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/1024.png");  sprintf(outputName, "./input/temp/o1024.png");  //noise= "./output/test2.png";
	//sprintf(filename, "./input/temp/1280x960org1.png");  sprintf(outputName, "./input/temp/o1280x960org1.png");  //noise= "./output/test2.png";

	

	
	
	//sprintf(filename, "./input/PR7.png");  //noise
		
	Mat  image;
	image = imread(filename, CV_LOAD_IMAGE_COLOR);
	
	if (!image.data)                              // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	// Display input 
	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	//imshow("Display window", image);                   // Show our image inside it.
	//waitKey(0);

	nWidth = image.cols;
	nHeight = image.rows; 

	Mat scrGray(nHeight, nWidth , CV_8U); 
	Mat dstGray(nHeight, nWidth, CV_8U);

	if (image.channels() == 3)
		cvtColor(image, scrGray, COLOR_BGR2GRAY);
	else if (image.channels() == 1)
		dstGray = image.clone();

	imwrite("./processDebug/input.png", scrGray);

	// calculate the padding area: 
	// right, bottom, 
	int win = 32*2; 
	int tempD = (nWidth%win);	
	int rightPad = nWidth%win;
	if (rightPad != 0) 
		rightPad = win - rightPad; 
	int bottomPad = nHeight%win; 
	if (bottomPad != 0)
		bottomPad = win - bottomPad; 

	int nWidthNew = nWidth + rightPad; 
	int nHeightNew = nHeight + bottomPad; 

	Mat scrGrayNew(nHeightNew, nWidthNew, CV_8U);
	Mat dstGrayNew(nHeightNew, nWidthNew, CV_8U);


	MakepadImageWin((unsigned char*)scrGray.data, (unsigned char*)scrGrayNew.data, nWidth, nHeight, rightPad, bottomPad, win, 1);
	imwrite("./processDebug/srcPad.png", scrGrayNew);


	//normalize((unsigned char *)dstGrayNew.data, (unsigned char *)dstGrayNew.data, 0, 255, CV_MINMAX);
	normalize(scrGrayNew, scrGrayNew, 0, 255, NORM_MINMAX, CV_8UC1);
	//normalize(dstGrayNew, dstGrayNew, 0, 255, CV_MINMAX);
	imwrite("./processDebug/srcPadNorm.png", scrGrayNew);

	//cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	//clahe->setClipLimit(2.0);
	//clahe->setTilesGridSize(Size(8,8)); 	
	//clahe->apply(scrGrayNew, scrGrayNew);
	//imwrite("./processDebug/srcPadNorm.png", scrGrayNew);


	BinarizeChar((unsigned char*)scrGrayNew.data, 
		(unsigned char*)dstGrayNew.data, nWidthNew, nHeightNew, 0);



	MakeCropPadImageWin((unsigned char*)dstGrayNew.data, (unsigned char*)dstGray.data, nWidth, nHeight, rightPad, bottomPad, win, 1);
	imwrite("./processDebug/outputPad.png", dstGrayNew);

	imwrite("./processDebug/output.png", dstGray);
	//imwrite("./output/test2.png", dstGray);
	imwrite(outputName, dstGray);
	
	
	return 0;

}

