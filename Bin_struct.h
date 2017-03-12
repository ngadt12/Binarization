/*
Files are composed as below    

			SS_struct.h                    
				  |                         
			 SS_function.h - SS_function.c  
				  |                         
			    test.c                     
*/

#include <stdio.h> 
#include <Windows.h>
#include <math.h>


#include <opencv\cv.h>		// Main OpenCV library.
#include <opencv\highgui.h>	// OpenCV functions for files and graphical windows.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

//#include "opencv/cv.h" // opencv library
//#include "opencv/highgui.h" // opencv library
#pragma comment(lib, "opencv_core310") // including opencv lib file 
#pragma comment(lib, "opencv_highgui310") // including opencv lib file 


//Image path
#define PATH_TESTDB "./TESTDB/indoor3.raw" //image path folder 
#define DEVICE_ORIENTATION 3


//patch path
#define PATCH_NUM_direction12 40
#define PATCH_W_direction12 32
#define PATCH_H_direction12 32
#define PATCH_NUM_direction34 44
#define PATCH_W_direction34 32
#define PATCH_H_direction34 32



// Image size
#define IMG_W 640 // image width 
#define IMG_H 480 // image height

#define resize_ratio 2// image resize ratio
#define IMG_w IMG_W/resize_ratio //resized image width
#define IMG_h IMG_H/resize_ratio //resized image height

#define TOTAL_SIZE (1.5*IMG_W*IMG_H)


#define F2_bin_1 32 // Color feature histogram bin 1 (We use 3dimensional color histogram)
#define F2_bin_2 32 // Color feature histogram bin 2
#define F2_bin_3 32 // Color feature histogram bin 3
#define F1_size IMG_w*IMG_h // F1 : Location probability
#define F2_size F2_bin_1*F2_bin_2*F2_bin_3 // F2 : Color probability 
#define F3_size 16 // F3 : Texture probability bin 
#define F3_r 1 // Texture feature radius (ex. LBP has 4 neighbor pixels)

typedef enum _HAND_DETECTION
{
	//Hand state

	HAND_DETECTION_INIT=0,
	HAND_DETECTION_SUCCESS=1,
	HAND_DETECTION_FAIL=-1,
}HAND_DETECTION;