#include "Bin_struct.h"


//Macro function 
#define     MAX_2(x, y)       (((x) > (y)) ? (x) : (y))
#define     MIN_2(x, y)       (((x) < (y)) ? (x) : (y)) //A function can find minimum number 

void Bin_ImgFlip(unsigned char *src, int w, int h);

void FileToArray(FILE *File, int *Array, int length);
int RawFile_length(FILE *infile, int W, int H);
void Bin_ImgResize_X2(unsigned char *src, const int src_width, const int src_height, const int dst_width, const int dst_height, unsigned char *dst);
void GetFrameFromRaw(unsigned char *RAW, int FrameNum, int W, int H, unsigned char *Y, unsigned char *U, unsigned char *V, unsigned char *U_, unsigned char *V_,int option);
void Bin_YUV2RGB(unsigned char *Y, unsigned char *U, unsigned char *V, const int width, const int height, unsigned char *dst);
// Int array to uchar array(type convert)
void INT2UCHAR(int *int_array, unsigned char *uchar_array, int length);
 
// OpenCV Ipl structure to uchar array//FPC O
void Bin_Ipl_to_Array(IplImage *src, unsigned char *dst, int length);

// Uchar array to OpenCV image struct(IplImage) //FPC O
void Bin_Array_to_Ipl(unsigned char *src, IplImage *dst, int length);
void Bin_Array_to_Ipl2(unsigned char *src, IplImage *dst, int length);	// grayLevel

// Array copy function//FPC O
void Bin_ArrayCopy(unsigned char *src, unsigned char *dst, int length);

// RGB to Gray converting function//FPC O
void Bin_RGB2GRAY(unsigned char *rgb, unsigned char *gray, int w , int h);

//Make array to zero //FPC O
void Bin_ArrayZero(unsigned char *src, int length);

//Swap to element function for quick sorting //FPC O
void swap(int *a, int *b);

//Quick sorting function for median filter //FPC O
void quick_sort(int *array, int start, int end);


void Bin_medianfilter(unsigned char *src, unsigned char *dst, int w, int h, int filter_size);


void Bin_medianfilter_rev20141207(unsigned char *src, unsigned char *dst, int w, int h);


//Finding element function for labeling //FPC O
int find( int set[], int x );

//image labeling function, binary image labeling //FPC O
int bwlabel(unsigned char* img, int n, int* labels, int w, int h);

//Save label area to array_area from labeled image  //FPC O
void Bin_label_area_to_array(int *label, int w, int h, int *array_area, int nobj, int *Hist_U, int *Hist_V, unsigned char *Y_, unsigned char *U_, unsigned char *V_);

//Ascending sorting //FPC O
void Bin_sorting_2D(int *array_, int *array_class, int array_num);

//Decending sorting //FPC O
void Bin_sorting_2D_reverse(int *array_, int *array_class, int array_num);

//Remove labeled image except the selected label //FPC O
void Bin_SelectLabel_region(int *Label, int label_num, unsigned char *dst, int w, int h);

//FPC O
int Bin_Finding_Hand_using_labeling(unsigned char *SRC, int w, int h, unsigned char *DST,  int *F3_p ,int *Hist_U, int* Hist_V, unsigned char *Y_, unsigned char *U_, unsigned char *V_, int *BOUND_BOX, unsigned char *patch12, int patch_num12, int patch_width12, int patch_height12,unsigned char *patch34, int patch_num34, int patch_width34, int patch_height34, int device_orientation, int *left_right_hand, int ROI_w_denominator, int ROI_w_max_numerator, int ROI_w_min_numerator,  int ROI_h_denominator, int ROI_h_max_numerator, int ROI_h_min_numerator);
//Hand region display function //FPC O
void Bin_HandRegion_Display(unsigned char *Hand, unsigned char *dst, int w, int h);

//Image resizing function //FPC O
void Bin_ImResize_Bilinear(unsigned char *src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int channel, unsigned char *dst);

// horizontal gradient function for edge
int Bin_Xgradient(unsigned char *src, int widthstep, int x, int y);

// Vertical gradient for sobel edge
int Bin_Ygradient(unsigned char *src, int widthstep, int x, int y);

//Diagonal direction gradient(45 degree)
int Bin_45gradient(unsigned char *src, int widthstep, int x, int y);

//Diagonal direction gradient(135 degree)
int Bin_135gradient(unsigned char *src, int widthstep, int x, int y);

// Sobel edge + Binary image function
void Bin_sobeledgeBinary(unsigned char *src, int w, int h, unsigned char *dst, int th);
void Bin_IfA255thenB01(unsigned char *src, unsigned char *dst, int w, int h);
// If image A at position (x,y) has pixel intensity 255, then make the intensity of image B at (x,y) to zero 
void Bin_IfA255thenB0(unsigned char *src, unsigned char *dst, int length);

// Texture function Local Binary Pattern has 8 neighbors pixels
int Bin_LBP8(unsigned char *src, int r, int widthstep, int i, int j);

// Texture function Local Binary Pattern has 4 neighbors pixels
int Bin_LBP4(unsigned char *src, int r, int widthstep, int i, int j);

// Hand region detection function
void Bin_HandCandidate(unsigned char *GRAY, unsigned char *BGR, unsigned char *Y_, unsigned char *U_, unsigned char *V_, int *F2_p, int *F2_n, int *F3_p, int *F3_n, int w, int h, int F2_HistBin_1, int F2_HistBin_2, int F2_HistBin_3, int F3_radius, unsigned char *dst, int *COLOR_THRESHOLD);

// Hand bounding box generation function
void Bin_Hand_Boundbox(unsigned char *hand, int w, int h, int *BOUNDBOX, const int HAND_STATE, int *region);
// Draw Bounding Box
void Bin_draw_horizontal_line(unsigned char *src, int ch_num, int width, int x1, int y1, int x2, int y2);
void Bin_draw_vertical_line(unsigned char *src, int ch_num, int width, int x1, int y1, int x2, int y2);
void Bin_draw_Rectangle(unsigned char *src, int ch_num, int img_width, int x1, int y1, int x2, int y2);

// Verify the HAND CANDIDATE == HAND or NOT
void Bin_HAND_VERIFY(unsigned char *hand, const int w, const int h, int *BOUNDBOX, int *HAND_class, const int HAND_STATE);

// Finger Pointing Vector Extraction
void SS_Pointing_Vector(unsigned char *hand, int w, int h, int orientation, int *BOUNDBOX, int *TIP, int *BASE, int *frame_storage);

// Region Of Interest(ROI) Seed Extraction
void Bin_ROI_SEED(unsigned char *hand, int w, int h, int finger_type, int *BOUNDBOX, int *TIP, int *BASE, int *SEED, int *ROI, const int ROI_length);
// Color Analysis using HSV Histogram function
void Bin_Color_Extraction(unsigned char *color_3ch, int img_width, int x1, int y1, int x2, int y2, int ROI_type, int *color_palette);

void Bin_A255B255C255toD255(unsigned char *A, unsigned char *B, unsigned char *C, unsigned char *D, int length);
void Bin_A255B255toD255(unsigned char *A, unsigned char *B, unsigned char *D, int length);


void Bin_IMG_rotation(unsigned char *src, unsigned char *dst, int nchannel, int width_src, int height_src, int width_dst, int height_dst);


void Bin_ROI(unsigned char *hand, int w, int h, int *TIP, int *BASE, int *ROI, const int ROI_extension_ratio);
void WHITE_BALANCE(unsigned char *src_BGR, const int src_w, const int src_h, unsigned char *dst_WB);


void Bin_Image_crop(unsigned char *src, int src_w, int src_h, unsigned char *dst, int dst_w, int dst_h,int x_st, int y_st, int x_en, int y_en);

int Bin_validation_using_matching(int matching_score, unsigned char *Y, unsigned char *hand_candidate, int hand_candidate_w, int hand_candidate_h, unsigned char *patch_samples, int width_of_patch, int height_of_patch, int patch_num, int *BOUND_BOX, int device_orientation, int *left_right_hand);

void Bin_fliplr(unsigned char *src, int w, int h);
void Bin_flipud(unsigned char *src, int w, int h);

void BinarizeChar(unsigned char* src, unsigned char* dst, int width, int height, int mode);
void Binarize(IplImage* srcIpl, IplImage* dstIpl, int width, int height, int mode);

int averageImage(unsigned char* src, int width, int height, int step);
void MinMaxImage(unsigned char* src, int width, int height, int step, int * MinMax);
void MakepadImage(unsigned char* src, unsigned char* dst, int width, int height, int win);
void MakepadImageWin(unsigned char* src, unsigned char* dst, int width, int height, int right, int bottom, int win, int channel);
void MakeCropPadImageWin(unsigned char* src, unsigned char* dst, int width, int height, int right, int bottom, int win, int channel);

void subtractImage(unsigned char* src, int* dst, int* dst2, int width, int height, int average);
void integralImage(int* src, int* dst, int width, int height);
int  LocalMeanStd(int* padImgG, int* padImg2G, int* LocalMean, int* LocalStd, int width, int height, int win);
void CalThreshold(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	);


void CalThresholdNew(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	);

void CalThresholdBGFG(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	);


void CalThresholdBGFG2(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	);

void FilterForeGround(unsigned char* srcdst, unsigned char* ref,
	int width, int height, 
	int win
);

int xGradient(unsigned char* src, int width, int height, int x, int y);
int yGradient(unsigned char* src, int width, int height, int x, int y);
int sobelFilter(unsigned char* src, unsigned char* dst, int width, int height);
int sobelImage(unsigned char* src, unsigned char* dst, unsigned char* kImg, int width, int height, int th1, int th2);

int signImage(unsigned char* src, unsigned char* meanImg, float* stdRImg,
	unsigned char* kImg, unsigned char* thr, int width, int height, unsigned char* thr1, unsigned char* thr2, float kVal);
void threshImg(unsigned char* src, unsigned char* thr, unsigned char* dst, int width, int height);

void WriteImage2Jpg(unsigned char* src, int width, int height, char* filename);

int range(int x, int y, int width, int height);