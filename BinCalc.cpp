#include "BinCalc.h"
#include <opencv\cv.h>		// Main OpenCV library.
#include <opencv\highgui.h>	// OpenCV functions for files and graphical windows.
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <cv.h>
//#include <highgui.h>

using namespace cv;
using namespace std;

#define MAX_INT 2147483647

void Bin_ImgFlip(unsigned char *src, int w, int h)
{
	int i = 0, j = 0;

	unsigned char *tmp = (unsigned char*)calloc(w*h, sizeof(unsigned char));
	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			tmp[i + j*w] = src[(w - 1 - i) + j*w];
		}
	}
	for (i = 0; i<w*h; i++)
		src[i] = tmp[i];
	free(tmp);

};

void FileToArray(FILE *File, int *Array, int length)
{
	int i = 0;
	for (i = 0; i < length; i++)
	{
		fscanf(File, "%d ", &Array[i]);
	}
};

int RawFile_length(FILE *infile, int W, int H)
{
	int i = 0, j = 0;
	int total_size = 0;
	int File_length_byte = 0;
	if (infile == NULL)
	{
		printf("File open error \n");
		return 0;
	}
	fseek(infile, 0, SEEK_END);
	File_length_byte = ftell(infile);
	rewind(infile);
	return File_length_byte;
};

void Bin_ImgResize_X2(unsigned char *src, const int src_width, const int src_height, const int dst_width, const int dst_height, unsigned char *dst)
{
	int i = 0, j = 0;
	for (i = 0; i < src_width; i++)
	{
		for (j = 0; j < src_height; j++)
		{
			int x = i * 2;
			int y = j * 2;
			dst[x + y*dst_width] = src[i + j*src_width];
			dst[x + (y + 1)*dst_width] = src[i + j*src_width];
			dst[x + 1 + y*dst_width] = src[i + j*src_width];
			dst[x + 1 + (y + 1)*dst_width] = src[i + j*src_width];
		}
	}
};

void GetFrameFromRaw(unsigned char *RAW, int FrameNum, int W, int H, unsigned char *Y, unsigned char *U, unsigned char *V, unsigned char *U_, unsigned char *V_, int option)
{
	int tmp = 0, i = 0, j = 0;
	int IMG_UV_W = W / 2;
	int IMG_UV_H = H / 2;
	int total_size = 1.5*W*H;
	unsigned char *Y_RAW = (unsigned char*)calloc(W*H, sizeof(unsigned char));
	unsigned char *U_RAW = (unsigned char*)calloc(0.25*W*H, sizeof(unsigned char));
	unsigned char *V_RAW = (unsigned char*)calloc(0.25*W*H, sizeof(unsigned char));
	//unsigned char *U_t = (unsigned char*)calloc(0.25*W*H, sizeof(unsigned char));
	//unsigned char *V_t = (unsigned char*)calloc(0.25*W*H, sizeof(unsigned char));
	for (i = total_size*FrameNum + 0; i < total_size*FrameNum + W*H; i++)
	{
		Y_RAW[tmp] = RAW[i];
		tmp++;
	}
	tmp = 0;
	for (i = total_size*FrameNum + W*H; i < total_size*FrameNum + 1.5*W*H; i += 2)
	{
		U_RAW[tmp] = RAW[i];
		tmp++;
	}
	tmp = 0;
	for (i = total_size*FrameNum + W*H + 1; i < total_size*FrameNum + 1.5*W*H + 1; i += 2)
	{
		V_RAW[tmp] = RAW[i];
		tmp++;
	}


	if (option==1)
	{
		tmp = 0;
		for (i = 0; i < H; i++)
		{
			for (j = 0; j <W; j++)
			{
				Y[j + i*W] = (unsigned char)Y_RAW[tmp];
				tmp++;
			}
		}
		tmp = 0;
		for (i = 0; i <IMG_UV_H; i++)
		{
			for (j = 0; j<IMG_UV_W ; j++)
			{
				U_[j + i*IMG_UV_W] = (unsigned char)U_RAW[tmp];
				tmp++;
			}
		}

		tmp = 0;
		for (i = 0; i <IMG_UV_H; i++)
		{
			for (j = 0; j<IMG_UV_W ; j++)
			{
				V_[j + i*IMG_UV_W] = (unsigned char)V_RAW[tmp];
				tmp++;
			}
		}

	}

	else if (option==-1)
	{
		tmp = 0;
		for (i = 0; i < W; i++)
		{
			for (j = H-1; j>=0; j--)
			{
				Y[i + j*W] = (unsigned char)Y_RAW[tmp];
				tmp++;
			}
		}
		tmp = 0;
		for (i = 0; i <IMG_UV_W; i++)
		{
			for (j = IMG_UV_H-1; j>=0 ; j--)
			{
				U_[i + j*IMG_UV_W] = (unsigned char)U_RAW[tmp];
				tmp++;
			}
		}

		tmp = 0;
		for (i = 0; i <IMG_UV_W; i++)
		{
			for (j = IMG_UV_H-1; j>=0 ; j--)
			{
				V_[i + j*IMG_UV_W] = (unsigned char)V_RAW[tmp];
				tmp++;
			}
		}

	}


	Bin_ImgResize_X2(V_, IMG_UV_W, IMG_UV_H, W, H, V);
	Bin_ImgResize_X2(U_, IMG_UV_W, IMG_UV_H, W, H, U);
	free(Y_RAW);
	free(U_RAW);
	free(V_RAW);

};

void Bin_YUV2RGB(unsigned char *Y, unsigned char *U, unsigned char *V, const int width, const int height, unsigned char *dst)
{
	int i;

	// Temporary storage is needed for int operation
	int *dst_temp;
	dst_temp = (int *)malloc(sizeof(int)* width * height * 3);


	// YUV 2 BGR conversion
	for (i = 0; i < width*height; i++)
	{
		// Channel R
		dst_temp[3 * i + 2] = ((298 * (Y[i] - 16) + 409 * (U[i] - 128) + 128) >> 8);

		// Channel G
		dst_temp[3 * i + 1] = ((298 * (Y[i] - 16) - 100 * (V[i] - 128) - 208 * (U[i] - 128) + 128) >> 8);

		// Channel B
		dst_temp[3 * i + 0] = ((298 * (Y[i] - 16) + 516 * (V[i] - 128) + 128) >> 8);

	}

	// Clamping a value to range of 0 to 255
	for (i = 0; i < width * height * 3; i++)
	{
		dst_temp[i] = (dst_temp[i] >= 255) ? 255 : dst_temp[i];
		dst_temp[i] = (dst_temp[i] <= 0) ? 0 : dst_temp[i];
		dst[i] = (unsigned char)dst_temp[i];
	}

	free(dst_temp);

};

// Int array to uchar array(type convert)
void INT2UCHAR(int *int_array, unsigned char *uchar_array, int length)
{
	//A function that can convert int type array to uchar type array;
	int i = 0;
	for (i = 0; i<length; i++)
		uchar_array[i] = (unsigned char)int_array[i];
};

void Bin_Ipl_to_Array(IplImage *src, unsigned char *dst, int length)
{
	// OpenCV Image struct is copied to uchar array
	// length is image size(ex. width x height)
	int i = 0;
	for (i = 0; i < length; i++)
	{
		unsigned char tmp= (unsigned char)src->imageData[i];
		dst[i] = tmp;
	}
};

void Bin_Array_to_Ipl(unsigned char *src, IplImage *dst, int length)
{
	// Copy uchar array to OpenCV image struct
	int i = 0;
	int y = 0;
#if 1
	for (i = 0; i < length; i++)
		dst->imageData[i] = (char)src[i];
#else
	for (y = 0; y < dst->height; y ++)
		memcpy(dst->imageData + y*dst->widthStep, src + y*dst->width, dst->width);
#endif
};


void Bin_Array_to_Ipl2(unsigned char *src, IplImage *dst, int length)
{
	// Copy uchar array to OpenCV image struct
	int i = 0;
	int y = 0;

	for (y = 0; y < dst->height; y ++)
		memcpy(dst->imageData + y*dst->widthStep, src + y*dst->width, dst->width);

};

void Bin_ArrayCopy(unsigned char *src, unsigned char *dst, int length)
{
	//Copy uchar array has length "length"
	int i = 0;
	for (i = 0; i < length; i++)
		dst[i] = src[i];
};

void Bin_RGB2GRAY(unsigned char *rgb, unsigned char *gray, int w, int h)
{
	// A function can convert to RGB to gray scale
	int i, j, R, G, B;
	for (i = 0; i < w; i++)
	{
		for (j = 0; j < h; j++)
		{
			B = (int)rgb[3 * j*w + 3 * i + 0];
			G = (int)rgb[3 * j*w + 3 * i + 1];
			R = (int)rgb[3 * j*w + 3 * i + 2];
			gray[j*w + i] = (unsigned char)((R + G + B) / 3);

		}
	}
};

void Bin_ArrayZero(unsigned char *src, int length)
{
	// A function that uchar array to zero
	memset(src, (unsigned char)0, sizeof(unsigned char)*length);
	//int i = 0;
	//for(i ; i < length ; i++)
	//	src[i] =(uchar)0;
};

void swap(int *a, int *b)
{
	// Swap two element function for quick sorting
	int tmp = *a;
	*a = *b;
	*b = tmp;
};

void quick_sort(int *array, int start, int end)
{
	// Quick sorting function for median filtering	
	int p, q;
	int mid = (start + end) / 2;
	int pivot = array[mid];
	if (start >= end) return;
	swap(&array[start], &array[mid]);
	p = start + 1;
	q = end;
	while (1)
	{
		while (array[p] <= pivot){ p++; }
		while (array[q]>pivot){ q--; }
		if (p>q) break;
		swap(&array[p], &array[q]);
	}
	swap(&array[start], &array[q]);
	quick_sort(array, start, q - 1);
	quick_sort(array, q + 1, end);
};
void Bin_medianfilter_rev20141207(unsigned char *src, unsigned char *dst, int w, int h)
{
	// Median filter function 
	// src : input image
	// dst : ouput image
	// w : width 
	// h : height
	int k = 0;
	int i = 1, j = 1;
	int *tmp = (int*)malloc(9 * sizeof(int));
	int tmp_sum = 0;
	int x=0;
	int y=0;

	int reference1  = 8;//((3*3)-1)/2;
	for(i =1 ; i< w -1 ; i++)
	{
		for(j =1 ; j< h -1 ; j++)
		{


			tmp_sum = tmp_sum+2*(int)src[(j - 1)*w + (i - 1)];
			tmp_sum = tmp_sum+3*(int)src[(j - 1)*w + (i + 0)];
			tmp_sum = tmp_sum+2*(int)src[(j - 1)*w + (i + 1)];
			tmp_sum = tmp_sum+3*(int)src[(j + 0)*w + (i - 1)];
			tmp_sum = tmp_sum+3*(int)src[(j + 0)*w + (i + 0)];
			tmp_sum = tmp_sum+3*(int)src[(j + 0)*w + (i + 1)];
			tmp_sum = tmp_sum+2*(int)src[(j + 1)*w + (i - 1)];
			tmp_sum = tmp_sum+3*(int)src[(j + 1)*w + (i + 0)];
			tmp_sum = tmp_sum+2*(int)src[(j + 1)*w + (i + 1)];


			if (tmp_sum>255 *reference1)
			{

				dst[j*w + i] = (unsigned char)255;
				tmp_sum=0;
			}
			else
			{
				dst[j*w + i] = (unsigned char)0;
				tmp_sum=0;
			}



		}
	}

	free(tmp);
};


void Bin_medianfilter(unsigned char *src, unsigned char *dst, int w, int h, int filter_size)
{
	// Median filter function 
	// src : input image
	// dst : ouput image
	// w : width 
	// h : height
	int k = 0;
	int i = 1, j = 1;
	//int *tmp = (int*)malloc(9 * sizeof(int));
	int tmp_sum = 0;
	int x=0;
	int y=0;
	int radius = (filter_size-1)/2;
	int reference1  = ((filter_size*filter_size)-1)/2;
	for(i =radius ; i< w -radius ; i++)
	{
		for(j =radius ; j< h -radius ; j++)
		{


			/*if((int)src[i+j*w]!=0)
			{*/
			//printf("\n새로운시작");
			for(x = -radius; x <= radius ;x++)
				for(y= -radius ; y <=radius ;y++)
				{
					tmp_sum = tmp_sum+(int)src[(j + y)*w + (i + x)];
					//printf("temp sum %d ", tmp_sum );

				}


				if (tmp_sum>255 *reference1)
				{

					dst[j*w + i] = (unsigned char)255;
					tmp_sum=0;
				}
				else
				{
					dst[j*w + i] = (unsigned char)0;
					tmp_sum=0;
				}
				//}


		}
	}


};

int find(int set[], int x)
{
	// finding function for labeling
	int r = x;
	while (set[r] != r)
		r = set[r];
	return r;
};

int bwlabel(unsigned char* img, int n, int* labels, int w, int h)
{
	// Image labeling function
	// img : input image
	// n : a number of neighbor pixels,( 4, or 8)
	// labels : output image(labeled image)
	// w: image width
	// h : image height
	int r = 0, c = 0, i = 0, j = 0;
	int nr = h;
	int nc = w;
	int total = nr * nc;
	// results
	//memset(labels, 0, total * sizeof(int));
	int nobj = 0;                               // number of objects found in image
	// other variables                             
	int* lset = (int*)malloc(total*sizeof(int));   // label table

	int ntable = 0;
	memset(lset, 0, total * sizeof(int));
	if (n != 4 && n != 8)
		n = 4;


	for (r = 0; r < nr; r++)
	{
		for (c = 0; c < nc; c++)
		{
			if (img[w*r + c])   // if A is an object
			{
				// get the neighboring pixels B, C, D, and E
				int B, C, D, E;
				if (c == 0)
					B = 0;
				else
					B = find(lset, labels[(r)*(nc)+c - 1]);
				//B = find( lset, ELEM1(labels, r, c - 1, nc) );
				if (r == 0)
					C = 0;
				else
					//C = find( lset, ELEM1(labels, r - 1, c, nc) );
					C = find(lset, labels[(r - 1)*(nc)+c]);
				if (r == 0 || c == 0)
					D = 0;
				else
					//D = find( lset, ELEM1(labels, r - 1, c - 1, nc) );
					D = find(lset, labels[(r - 1)*(nc)+c - 1]);
				if (r == 0 || c == nc - 1)
					E = 0;
				else
					//E = find( lset, ELEM1(labels, r - 1, c + 1, nc) );
					E = find(lset, labels[(r - 1)*(nc)+c + 1]);
				if (n == 4)
				{
					// apply 4 connectedness
					if (B && C)
					{        // B and C are labeled
						if (B == C)
							//ELEM1(labels, r, c, nc) = B;
							labels[(r)*(nc)+c] = B;
						else {
							lset[C] = B;
							//ELEM1(labels, r, c, nc) = B;
							labels[r*nc + c] = B;
						}
					}
					else if (B)             // B is object but C is not
						//ELEM1(labels, r, c, nc) = B;
						labels[r*nc + c] = B;
					else if (C)               // C is object but B is not
						//ELEM1(labels, r, c, nc) = C;
						labels[r*nc + c] = C;
					else
					{                      // B, C, D not object - new object
						//   label and put into table
						ntable++;
						//ELEM1(labels, r, c, nc) = lset[ ntable ] = ntable;
						labels[r*nc + c] = lset[ntable] = ntable;

					}
				}
				else if (n == 6)
				{
					// apply 6 connected ness
					if (D)                    // D object, copy label and move on
						//ELEM1(labels, r, c, nc) = D;
						labels[r*nc + c] = D;
					else if (B && C)
					{        // B and C are labeled
						if (B == C)
							//ELEM1(labels, r, c, nc) = B;
							labels[r*nc + c] = B;
						else
						{
							int tlabel = MIN_2(B, C);
							lset[B] = tlabel;
							lset[C] = tlabel;
							//ELEM1(labels, r, c, nc) = tlabel;
							labels[r*nc + c] = tlabel;

						}
					}
					else if (B)             // B is object but C is not
						//ELEM1(labels, r, c, nc) = B;
						labels[r*nc + c] = B;
					else if (C)               // C is object but B is not
						//ELEM1(labels, r, c, nc) = C;
						labels[r*nc + c] = C;
					else
					{                      // B, C, D not object - new object
						//   label and put into table
						ntable++;
						//ELEM1(labels, r, c, nc) = lset[ ntable ] = ntable;
						labels[r*nc + c] = lset[ntable] = ntable;
					}
				}
				else if (n == 8)
				{
					// apply 8 connectedness
					if (B || C || D || E)
					{
						int tlabel = B;
						if (B)
							tlabel = B;
						else if (C)
							tlabel = C;
						else if (D)
							tlabel = D;
						else if (E)
							tlabel = E;
						//ELEM1(labels, r, c, nc) = tlabel;
						labels[r*nc + c] = tlabel;

						if (B && B != tlabel)
							lset[B] = tlabel;
						if (C && C != tlabel)
							lset[C] = tlabel;
						if (D && D != tlabel)
							lset[D] = tlabel;
						if (E && E != tlabel)
							lset[E] = tlabel;
					}
					else
					{
						//   label and put into table
						ntable++;
						//ELEM1(labels, r, c, nc) = lset[ ntable ] = ntable;
						labels[r*nc + c] = lset[ntable] = ntable;

					}
				}
			}
			else
			{
				//ELEM1(labels, r, c, nc) = 0;      // A is not an object so leave it
				labels[r*nc + c] = 0;
			}
		}
	}
	// consolidate component table
	for (i = 0; i <= ntable; i++)
		lset[i] = find(lset, i);
	// run image through the look-up table
	for (r = 0; r < nr; r++)
		for (c = 0; c < nc; c++)
			//ELEM1(labels, r, c, nc) = lset[ ELEM1(labels, r, c, nc) ];
			labels[r*nc + c] = lset[labels[r*nc + c]];


	// count up the objects in the image
	for (i = 0; i <= ntable; i++)
		lset[i] = 0;
	for (r = 0; r < nr; r++)
		for (c = 0; c < nc; c++)
			//lset[ ELEM1(labels, r, c, nc) ]++;
			lset[labels[r*nc + c]]++;
	// number the objects from 1 through n objects
	nobj = 0;
	lset[0] = 0;
	for (i = 1; i <= ntable; i++)
		if (lset[i] > 0)
			lset[i] = ++nobj;
	// run through the look-up table again
	for (r = 0; r < nr; r++)
		for (c = 0; c < nc; c++)
			//ELEM1(labels, r, c, nc) = lset[ ELEM1(labels, r, c, nc) ];
			labels[r*nc + c] = lset[labels[r*nc + c]];

	//
	free(lset);
	return nobj;
};

void Bin_label_area_to_array(int *label, int w, int h, int *array_area, int nobj,int *Hist_U, int *Hist_V, unsigned char *Y_, unsigned char *U_, unsigned char *V_)
{
	// label : labeled image
	// w: width
	// h : height
	// array_area : Array for saving area value of each labeled cluster
	// nobj : total labeld number
	int i = 0, j = 0, k = 0;
	int *array_area_not_weight = (int*)calloc(nobj, sizeof(int));

	for (i = 0; i < w*h; i++)
	{
		for (k = 0; k<nobj; k++)
		{
			if (label[i] == k + 1)
			{
				array_area_not_weight[k]=array_area_not_weight[k]+1;
			}
		}

	}
	for (i = 0; i < w; i++)
	{
		for (j = 0; j < h; j++)
		{
			for (k = 0; k<nobj; k++)
			{
				if (label[i + j*w] == k + 1 && array_area_not_weight[k]>1000)
				{
					int tmp_Y =(int)Y_[i+j*w];
					int tmp_U =(int)U_[i+j*w];
					int tmp_V =(int)V_[i+j*w];

					array_area[k]=array_area[k]+(int)(Hist_U[tmp_U]*Hist_V[tmp_V]);
				}
			}
		}
	}
	free(array_area_not_weight);

};

void Bin_sorting_2D(int *array_, int *array_class, int array_num)
{
	// Ascending sort regard with array_
	// We sorting the array_area and labeled class simultaneously

	int j = 0, i = 0;
	for (i = 1; i < array_num; i++)
	{
		int tmp = array_[i];
		int tmp_class = array_class[i];
		for (j = i - 1; j >= 0; j--)
		{
			if (array_[j]>tmp)
			{
				array_[j + 1] = array_[j];
				array_class[j + 1] = array_class[j];
			}
			else
				break;
		}
		array_[j + 1] = tmp;
		array_class[j + 1] = tmp_class;

	}
};

void Bin_sorting_2D_reverse(int *array_, int *array_class, int array_num)
{
	// Reverse sorted array

	int *tmp_data = (int*)malloc(sizeof(int)*array_num);
	int *tmp_class = (int*)malloc(sizeof(int)*array_num);
	int i = 0;
	for (i = 0; i < array_num; i++)
	{
		tmp_data[array_num - i - 1] = array_[i];
		tmp_class[array_num - i - 1] = array_class[i];
	}

	for (i = 0; i < array_num; i++)
	{
		array_[i] = tmp_data[i];
		array_class[i] = tmp_class[i];

	}
	free(tmp_class);
	free(tmp_data);
};

void Bin_SelectLabel_region(int *Label, int label_num, unsigned char *dst, int w, int h)
{
	// Label : Labeled image
	// label_class : An array contains labeld number(sorted with area) 
	// order : A "order"th array element of sorted array(label_class) 
	// dst : Image contains just we select("order"th array) labeled
	int i = 0;
	for (i = 0; i < w*h; i++)
		dst[i] = (Label[i] == label_num) ? 255 : 0;
};

int Bin_Finding_Hand_using_labeling(unsigned char *SRC, int w, int h, unsigned char *DST,  int *F3_p ,int *Hist_U, int* Hist_V, unsigned char *Y_, unsigned char *U_, unsigned char *V_, int *BOUND_BOX, unsigned char *patch12, int patch_num12, int patch_width12, int patch_height12,unsigned char *patch34, int patch_num34, int patch_width34, int patch_height34, int device_orientation, int *left_right_hand, int ROI_w_denominator, int ROI_w_max_numerator, int ROI_w_min_numerator,  int ROI_h_denominator, int ROI_h_max_numerator, int ROI_h_min_numerator)
{
	//Find hand candidate from image
	//SRC : Hand Probability map from "SS_HandCandidate" function
	//w : width
	//h: height
	// DST : output image(Hand region)

	// Labeling image memory allocation

	int *LABEL = (int*)calloc(w*h, sizeof(int));
	//IplImage *src = cvCreateImage (cvSize(w, h), IPL_DEPTH_8U, 1);
	// nobj : a number of labeled number.
	int nobj = bwlabel(SRC, 4, LABEL, w, h);
	int i = 0;
	int *L_area = (int*)calloc(nobj, sizeof(int));
	int *L_class = (int*)calloc(nobj, sizeof(int));
	int CANDIDATE_NUM=0;
	//IplImage *display = cvCreateImage(cvSize(IMG_w, IMG_h), IPL_DEPTH_8U, 1);

	int tmp=0;
	for (i = 0; i < nobj; i++)
	{
		L_area[i] = 0;
		L_class[i] = i + 1;
	}
	Bin_label_area_to_array(LABEL, w, h, L_area, nobj, Hist_U, Hist_V, Y_, U_, V_);
	Bin_sorting_2D(L_area, L_class, nobj);
	Bin_sorting_2D_reverse(L_area, L_class, nobj);
	//for(i = 0 ; i < nobj ; i++)if(L_area[i]>=1000000)printf("\nL_class : %d  L_area : %d", L_class[i], L_area[i]);
	for(i = 0 ; i < nobj ; i++)if(L_area[i]>=1000000)CANDIDATE_NUM++;
	//printf("%d\n",  CANDIDATE_NUM);
	if(CANDIDATE_NUM>=1)
	{
		int *MATCHING_SCORE = (int*)calloc(CANDIDATE_NUM,sizeof(int));
		int *MATCHING_SCORE_label = (int*)calloc(CANDIDATE_NUM, sizeof(int));
		for(tmp = 0 ; tmp < CANDIDATE_NUM ; tmp++)MATCHING_SCORE_label[tmp]=L_class[tmp];

		for(i=0 ; i<CANDIDATE_NUM ; i++)
		{
			int matching_score=0;
			int *region=(int*)calloc(1, sizeof(int));
			Bin_ArrayZero(DST, IMG_w*IMG_h);
			Bin_SelectLabel_region(LABEL, L_class[i], DST, IMG_w, IMG_h);
			// Hand Region Bounding Box Generation
			BOUND_BOX[0]=0;
			BOUND_BOX[1]=0;
			BOUND_BOX[2]=0;
			BOUND_BOX[3]=0;
			Bin_Hand_Boundbox(DST, IMG_w, IMG_h, BOUND_BOX,1,region );

			if (    ((int)(BOUND_BOX[2]-BOUND_BOX[0]) <= (int)(IMG_w*ROI_w_max_numerator/ROI_w_denominator))
				&& 
				((int)(BOUND_BOX[3]-BOUND_BOX[1]) <= (int)(IMG_h*ROI_h_max_numerator/ROI_h_denominator)) 
				&&
				((int)(BOUND_BOX[2]-BOUND_BOX[0]) >= (int)(IMG_w*ROI_w_min_numerator/ROI_w_denominator)) 
				&&
				((int)(BOUND_BOX[3]-BOUND_BOX[1]) >= (int)(IMG_h*ROI_h_min_numerator/ROI_h_denominator))   
				&&
				(region[0]< (int)(0.9*((BOUND_BOX[2]-BOUND_BOX[0])*(BOUND_BOX[3]-BOUND_BOX[1]))))
				&&(region[0]> (int)(0.1*((BOUND_BOX[2]-BOUND_BOX[0])*(BOUND_BOX[3]-BOUND_BOX[1]))))
				)
			{
				if(device_orientation==1 ||  device_orientation==2)
					matching_score = Bin_validation_using_matching(matching_score, Y_, DST, w,h,patch12,patch_width12, patch_height12,patch_num12,BOUND_BOX,device_orientation, left_right_hand);
				else if(device_orientation==3 ||  device_orientation==4)
					matching_score = Bin_validation_using_matching(matching_score,Y_,DST, w,h,patch34,patch_width34, patch_height34,patch_num34,BOUND_BOX,device_orientation, left_right_hand);
				MATCHING_SCORE[i]=matching_score;
			}
			else
				MATCHING_SCORE[i]=INT_MAX;
			free(region);
		}
		Bin_sorting_2D(MATCHING_SCORE, MATCHING_SCORE_label, CANDIDATE_NUM);
		//SS_sorting_2D_reverse(L_area, L_class, nobj);
		//for(i = 0 ; i < CANDIDATE_NUM ; i++)printf("Label class : %d  MATCHING_SCORE : %d\n", MATCHING_SCORE_label[i], MATCHING_SCORE[i]);
		if(MATCHING_SCORE[0]<75000)
		{
			int *region=(int*)calloc(1, sizeof(int));
			Bin_SelectLabel_region(LABEL, MATCHING_SCORE_label[0], DST, IMG_w, IMG_h);
			Bin_Hand_Boundbox(DST, IMG_w, IMG_h, BOUND_BOX,1,region );
			//SS_Array_to_Ipl(DST, display, IMG_w*IMG_h);cvShowImage("display", display);
			free(region);
			free(L_area);
			free(L_class);
			free(LABEL);
			free(MATCHING_SCORE);
			free(MATCHING_SCORE_label);

			return 1;
		}
		else
		{
			BOUND_BOX[0]=0;
			BOUND_BOX[1]=0;
			BOUND_BOX[2]=0;
			BOUND_BOX[3]=0;

			Bin_ArrayZero(DST, IMG_w*IMG_h);
			free(L_area);
			free(L_class);
			free(LABEL);
			free(MATCHING_SCORE);
			free(MATCHING_SCORE_label);
			return -1;
		}
	}
	//			
	//			if (    ((int)(BOUND_BOX[2]-BOUND_BOX[0]) <= (int)(IMG_w*ROI_w_max_numerator/ROI_w_denominator))
	//																						&& 
	//					((int)(BOUND_BOX[3]-BOUND_BOX[1]) <= (int)(IMG_h*ROI_h_max_numerator/ROI_h_denominator)) 
	//																						&&
	//				    ((int)(BOUND_BOX[2]-BOUND_BOX[0]) >= (int)(IMG_w*ROI_w_min_numerator/ROI_w_denominator)) 
	//																						&&
	//				    ((int)(BOUND_BOX[3]-BOUND_BOX[1]) >= (int)(IMG_h*ROI_h_min_numerator/ROI_h_denominator))   
	//																						&&
	//					(region[0]< (int)(0.9*((BOUND_BOX[2]-BOUND_BOX[0])*(BOUND_BOX[3]-BOUND_BOX[1]))))
	//				//&&(region[0]> (int)(0.1*((BOUND_BOX[2]-BOUND_BOX[0])*(BOUND_BOX[3]-BOUND_BOX[1]))))
	//				)
	//			{
	//				
	//#if 0
	//				free(L_area);
	//				free(L_class);
	//				free(LABEL);
	//				return 1;
	//#else				
	//				int check_=0;
	//				
	//				if(device_orientation==1 ||  device_orientation==2)
	//					check_ = SS_validation_using_matching(MATCHING_SCORE, MATCHING_SCORE_label, Y_, DST, w,h,patch12,patch_width12, patch_height12,patch_num12,BOUND_BOX,device_orientation, left_right_hand);
	//				else if(device_orientation==3 ||  device_orientation==4)
	//					check_ = SS_validation_using_matching(MATCHING_SCORE, MATCHING_SCORE_label,Y_,DST, w,h,patch34,patch_width34, patch_height34,patch_num34,BOUND_BOX,device_orientation, left_right_hand);
	//
	//				if(check_!=1 &&L_area[i+1]>1000000)
	//				{
	//					continue;
	//				}
	//				else if(check_!=1 && L_area[i+1]<=1000000)
	//				{
	//					SS_ArrayZero(DST, IMG_w*IMG_h);
	//					left_right_hand[0]=0;
	//					free(L_area);
	//					free(L_class);
	//					free(LABEL);
	//					return -1;
	//
	//				}
	//
	//				else
	//				{
	//					free(L_area);
	//					free(L_class);
	//					free(LABEL);
	//					return 1;
	//				}
	//#endif
	//			}
	//			else
	//			{
	//				left_right_hand[0]=0;
	//				SS_ArrayZero(DST, IMG_w*IMG_h);
	//				continue;
	//				free(L_area);
	//				free(L_class);
	//				free(LABEL);
	//				return -1;
	//			}
	//
	//		}
	//		else
	//		{
	//			left_right_hand[0]=0;
	//			//printf("HAND detection fail \n");
	//			SS_ArrayZero(DST, IMG_w*IMG_h);
	//			free(L_area);
	//			free(L_class);
	//			free(LABEL);
	//			return -1;
	//		}
	//
	//}

};

void Bin_HandRegion_Display(unsigned char *Hand, unsigned char *dst, int w, int h)
{
	// Hand region display function
	// Hand : hand region(1ch) 
	// dst : original image

	int i = 0, j = 0;
	for (i = 0; i < w; i++)
	{
		for (j = 0; j < h; j++)
		{
			if (Hand[i + j*w] != (unsigned char)0)
			{
				dst[0 + 3 * i + j * 3 * w] = (unsigned char)230;
				dst[1 + 3 * i + j * 3 * w] = (unsigned char)30;
				dst[2 + 3 * i + j * 3 * w] = (unsigned char)30;
			}
		}
	}
};

void Bin_ImResize_Bilinear(unsigned char *src, const int src_width, const int src_height, const int dst_width, const int dst_height, const int channel, unsigned char *dst)
{
	int i, j;
	int src_widthStep;

	int ratio_X, ratio_Y;
	int axis_X, axis_Y;

	int axis_X_int, axis_Y_int;

	int bilinear_step_1, bilinear_step_2, bilinear_step_3;
	int bilinear_step_1_B, bilinear_step_2_B, bilinear_step_3_B, bilinear_step_1_G, bilinear_step_2_G, bilinear_step_3_G, bilinear_step_1_R, bilinear_step_2_R, bilinear_step_3_R;


	// Calculate the resize ratio
	ratio_X = src_width / dst_width;
	ratio_Y = src_height / dst_height;

	// Calculate the matching point axis (Grayscale)
	if (channel == 1)
	{
		for (i = 0; i<dst_width; i++)
		{
			for (j = 0; j<dst_height; j++)
			{
				axis_X = i * ratio_X;
				axis_Y = j * ratio_Y;

				axis_X_int = axis_X;
				axis_Y_int = axis_Y;

				bilinear_step_1 = ((axis_X_int + 1) - axis_X)*(src[(axis_Y_int*src_width) + axis_X_int]) + (axis_X - (axis_X_int))*(src[(axis_Y_int*src_width) + axis_X_int + 1]);
				bilinear_step_2 = ((axis_X_int + 1) - axis_X)*(src[((axis_Y_int + 1)*src_width) + axis_X_int]) + (axis_X - (axis_X_int))*(src[((axis_Y_int + 1)*src_width) + axis_X_int + 1]);
				bilinear_step_3 = ((axis_Y_int + 1) - axis_Y)*(bilinear_step_1)+(axis_Y - (axis_Y_int))*(bilinear_step_2);

				(bilinear_step_3 > 255) ? 255 : bilinear_step_3;
				(bilinear_step_3 < 0) ? 0 : bilinear_step_3;

				dst[j*dst_width + i] = (unsigned char)bilinear_step_3;
			}
		}
	}


	// Calculate the matching point axis (Color)
	if (channel == 3)
	{
		src_widthStep = src_width * channel;

		for (i = 0; i<dst_width; i++)
		{
			for (j = 0; j<dst_height; j++)
			{
				axis_X = i * ratio_X;
				axis_Y = j * ratio_Y;

				axis_X_int = axis_X;
				axis_Y_int = axis_Y;

				bilinear_step_1_B = ((axis_X_int + 1) - axis_X)*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int)]) + (axis_X - (axis_X_int))*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int) + 3]);
				bilinear_step_2_B = ((axis_X_int + 1) - axis_X)*src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int)] + (axis_X - (axis_X_int))*src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int) + 3];
				bilinear_step_3_B = ((axis_Y_int + 1) - axis_Y)*(bilinear_step_1_B)+(axis_Y - (axis_Y_int))*(bilinear_step_2_B);
				dst[(j*dst_width*channel) + 3 * i] = (unsigned char)bilinear_step_3_B;

				bilinear_step_1_G = ((axis_X_int + 1) - axis_X)*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int) + 1]) + (axis_X - (axis_X_int))*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int) + 4]);
				bilinear_step_2_G = ((axis_X_int + 1) - axis_X)*(src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int) + 1]) + (axis_X - (axis_X_int))*(src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int) + 4]);
				bilinear_step_3_G = ((axis_Y_int + 1) - axis_Y)*(bilinear_step_1_G)+(axis_Y - (axis_Y_int))*(bilinear_step_2_G);
				dst[(j*dst_width*channel) + 3 * i + 1] = (unsigned char)bilinear_step_3_G;

				bilinear_step_1_R = ((axis_X_int + 1) - axis_X)*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int) + 2]) + (axis_X - (axis_X_int))*(src[(axis_Y_int*src_widthStep) + (3 * axis_X_int) + 5]);
				bilinear_step_2_R = ((axis_X_int + 1) - axis_X)*(src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int) + 2]) + (axis_X - (axis_X_int))*(src[((axis_Y_int + 1)*src_widthStep) + (3 * axis_X_int) + 5]);
				bilinear_step_3_R = ((axis_Y_int + 1) - axis_Y)*(bilinear_step_1_R)+(axis_Y - (axis_Y_int))*(bilinear_step_2_R);
				dst[(j*dst_width*channel) + 3 * i + 2] = (unsigned char)bilinear_step_3_R;
			}
		}
	}

};

int Bin_Xgradient(unsigned char *src, int widthstep, int x, int y)
{
	//Horizontal gradient fuction for sobel edge function
	//Calculate horizontal gradient at image (x,y)
	int tmp_0 = (int)src[widthstep*(y - 1) + x - 1];
	int tmp_1 = 3 * (int)src[widthstep*(y)+x - 1];
	int tmp_2 = (int)src[widthstep*(y + 1) + x - 1];
	int tmp_3 = -(int)src[widthstep*(y - 1) + x + 1];
	int tmp_4 = -3 * (int)src[widthstep*y + x + 1];
	int tmp_5 = -(int)src[widthstep*(y + 1) + x + 1];
	int tmp_ = tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5;
	tmp_ = tmp_ >= 0 ? tmp_ : -tmp_;
	return tmp_;
};
int Bin_Ygradient(unsigned char *src, int widthstep, int x, int y)
{
	int tmp_0 = (int)src[widthstep*(y - 1) + x - 1];
	int tmp_1 = 3 * (int)src[widthstep*(y - 1) + x];
	int tmp_2 = (int)src[widthstep*(y - 1) + x + 1];
	int tmp_3 = -(int)src[widthstep*(y + 1) + x - 1];
	int tmp_4 = -3 * (int)src[widthstep*(y + 1) + x];
	int tmp_5 = -(int)src[widthstep*(y + 1) + x + 1];
	int tmp_ = tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5;
	tmp_ = tmp_ >= 0 ? tmp_ : -tmp_;
	return tmp_;

};
int Bin_45gradient(unsigned char *src, int widthstep, int x, int y)
{
	int tmp_0 = (int)src[widthstep*(y)+x - 1];
	int tmp_1 = 2 * (int)src[widthstep*(y + 1) + x - 1];
	int tmp_2 = (int)src[widthstep*(y + 1) + x];
	int tmp_3 = -(int)src[widthstep*(y - 1) + x];
	int tmp_4 = -2 * (int)src[widthstep*(y - 1) + x + 1];
	int tmp_5 = -(int)src[widthstep*(y)+x + 1];
	int tmp_ = tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5;
	tmp_ = tmp_ >= 0 ? tmp_ : -tmp_;
	return tmp_;

};
int Bin_135gradient(unsigned char *src, int widthstep, int x, int y)
{
	int tmp_0 = (int)src[widthstep*(y - 1) + x];
	int tmp_1 = 2 * (int)src[widthstep*(y - 1) + x - 1];
	int tmp_2 = (int)src[widthstep*(y)+x - 1];
	int tmp_3 = -(int)src[widthstep*(y)+x - 1];
	int tmp_4 = -2 * (int)src[widthstep*(y + 1) + x + 1];
	int tmp_5 = -(int)src[widthstep*(y + 1) + x];
	int tmp_ = tmp_0 + tmp_1 + tmp_2 + tmp_3 + tmp_4 + tmp_5;
	tmp_ = tmp_ >= 0 ? tmp_ : -tmp_;
	return tmp_;

};

void Bin_sobeledgeBinary(unsigned char *src, int w, int h, unsigned char *dst, int th)
{
	// Sobel edge function
	//src : input
	// w : width
	//h : height
	// dst : output(edge image)
	int i, j, tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;
	int indexJ = w; 
	for(j = 1 ; j < h-1 ; j++) 
	{
		for(i = 1 ; i < w-1 ; i++)
		{
			tmp0 = Bin_Xgradient(src, w, i, j);
			tmp1 = Bin_Ygradient(src, w, i, j);
			tmp2 = Bin_45gradient(src, w, i, j);
			tmp3 = Bin_135gradient(src, w, i, j);
			tmp4 = MAX_2(MAX_2(MAX_2(tmp0,tmp1), tmp2), tmp3);
			dst[indexJ+i]=(unsigned char)tmp4>(unsigned char)th?255:0;
		}
		indexJ += w; 
	}
};
void Bin_IfA255thenB01(unsigned char *src, unsigned char *dst, int w, int h)
{
	int i= 0, j=0;
	for( i = 1 ; i < w-1 ;i++ )
	{
		for(j=1 ; j < h-1 ;j++)
		{
			if((int)src[i+j*w]!=0)
			{
				dst[i+j*w]=(unsigned char)0;
				dst[i-1+j*w]=(unsigned char)0;
				//dst[i+1+j*w]=(unsigned char)0;
				dst[i+(j-1)*w]=(unsigned char)0;
				//dst[i+(j+1)*w]=(unsigned char)0;
			}
		}
	}

};



void Bin_IfA255thenB0(unsigned char *src, unsigned char *dst, int length)
{
	// If image A at position (x,y) has pixel intensity 255, then make the intensity of image B at (x,y) to zero 
	int i;
	for(i = 0; i < length; i++)
		dst[i] = src[i] == (unsigned char)255 ? 0 : dst[i];
};

int Bin_LBP8(unsigned char *src, int r, int widthstep, int i, int j)
{
	//Calculate a local binary pattern value (LBP has 8 neightbor pixels)
	// src : input image 
	// r: LBP radius
	// widthstep : image width
	// i : position x
	// j : position y
	int t0 = (int)src[i + widthstep*j];
	int t1 = t0 - (int)src[i - r + widthstep*(j - r)]>0 ? 1 : 0;
	int t2 = t0 - (int)src[i + widthstep*(j - r)]>0 ? 1 : 0;
	int t3 = t0 - (int)src[i + r + widthstep*(j - r)]>0 ? 1 : 0;
	int t4 = t0 - (int)src[i + r + widthstep*(j)]>0 ? 1 : 0;
	int t5 = t0 - (int)src[i + r + widthstep*(j + r)]>0 ? 1 : 0;
	int t6 = t0 - (int)src[i + widthstep*(j + r)]>0 ? 1 : 0;
	int t7 = t0 - (int)src[i - r + widthstep*(j + r)]>0 ? 1 : 0;
	int t8 = t0 - (int)src[i - r + widthstep*(j)]>0 ? 1 : 0;
	return t1 * 1 + t2 * 2 + t3 * 4 + t4 * 8 + t5 * 16 + t6 * 32 + t7 * 64 + t8 * 128;
};
int Bin_LBP4(unsigned char *src, int r, int widthstep, int i, int j)
{
	//Calculate a local binary pattern value (LBP has 4 neightbor pixels)
	// src : input image 
	// r: LBP radius
	// widthstep : image width
	// i : position x
	// j : position y
	int index = i+widthstep*j;
	int index1 = widthstep*r; 

	int t0= (int)src[index];
	int t2 = t0-(int)src[index-index1]>0? 1:0;
	int t4 = t0-(int)src[index+r]>0? 1: 0;
	int t6 = t0-(int)src[index+index1]>0? 1: 0;
	int t8 = t0-(int)src[index-r]>0? 1: 0;
	return t2*1+t4*2+t6*4+t8*8;
}

void Bin_HandCandidate(unsigned char *GRAY, unsigned char *BGR, unsigned char *Y_, unsigned char *U_, unsigned char *V_, int *F2_p, int *F2_n, int *F3_p, int *F3_n, int w, int h, int F2_HistBin_1, int F2_HistBin_2, int F2_HistBin_3, int F3_radius, unsigned char *dst, int *COLOR_THRESHOLD)
{
	// Finding Hand candidate using the multi modal probability
	// GRAY : gray scale image(input)
	// BGR : color image(input)
	// F1_p : An array contains positive F1 data(hand location probability) 
	// F1_n : An array contains negative F1 data(non-hand location probability) 
	// F2_p : An array contains positive F2 data(hand color probability) 
	// F2_n : An array contains negative F2 data(non-hand color probability) 
	// F3_p : An array contains positive F3 data(hand texture,LBP probability) 
	// F3_n : An array contains negative F3 data(non-hand texture,LBP probability) 
	// w: width
	// h : height
	// F2_bin_1 : 1st bin size of color probability map
	// F2_bin_2 : 2nd bin size of color probability map
	// F2_bin_3 : 3rd bin size of color probability map
	// dst : output hand region probability map

	int i, j;
	int w0, w1;
	int B_r = (255/F2_HistBin_1)+1;
	int G_r = (255/F2_HistBin_2)+1;
	int R_r = (255/F2_HistBin_3)+1;
	int index = 0, index1; 
	int B, G, R, t0;
	int F2_HistBin_2x3 = F2_HistBin_2*F2_HistBin_3;
	int tempLog;
	//printf("hhhh%d", COLOR_THRESHOLD[0]);
	for( j = F3_radius ; j < h - F3_radius ; j++) 
	{
		for(i=F3_radius ; i < w-F3_radius ; i++)
		{   
			if((int)(Y_[i+j*w])>COLOR_THRESHOLD[0] &&  (int)(Y_[i+j*w]) <COLOR_THRESHOLD[1] 
			&& (int)(U_[i+j*w])>COLOR_THRESHOLD[2] &&  (int)(U_[i+j*w]) <COLOR_THRESHOLD[3] 
			&& (int)(V_[i+j*w])>COLOR_THRESHOLD[4] &&  (int)(V_[i+j*w]) <COLOR_THRESHOLD[5]  
			/*&& (int)(BGR[i*3+0+j*w*3]>15)
			&& (int)(BGR[i*3+1+j*w*3]>30)
			&& (int)(BGR[i*3+2+j*w*3]>80)*/
			//&& (int)(BGR[i*3+2+j*w*3]-BGR[i*3+1+j*w*3]>15)

			)
			{
				//Color
				index = (j*w+i)*3;// j*3*w+3*i;
				index1 = j*w+i; 
				B  = (int)(BGR[index+0]/B_r);
				G = (int)(BGR[index+1]/G_r);
				R = (int)(BGR[index+2]/R_r);

				//Texture 
				t0= Bin_LBP4(GRAY, F3_r, IMG_w, i, j);

				tempLog = B*F2_HistBin_2x3 + G*F2_HistBin_3 + R;
				//w0 = weight1*log((double)F1_p[index1])+weight2*log((double)F2_p[tempLog])+weight3*log((double)F3_p[t0]);
				//w1 = weight1*log((double)F1_n[index1])+weight2*log((double)F2_n[tempLog])+weight3*log((double)F3_n[t0]);

				w0 = 1*log((double)F2_p[tempLog])+(int)1*log((double)F3_p[t0]);
				w1 = 1*log((double)F2_n[tempLog])+(int)1*log((double)F3_n[t0]);

				if(w0>w1*0.7)
					dst[index1]=255;
			}
		}
	}
}

void Bin_Hand_Boundbox(unsigned char *hand, int w, int h, int *BOUNDBOX, const int HAND_STATE, int *region)
{
	// hand : hand candidate image
	// w : width
	// h : height
	// BOUNDBOX[0] : BOUND_start.x,  BOUNDBOX[1] : BOUND_start.y
	// BOUNDBOX[2] : BOUND_end.x,    BOUNDBOX[3] : BOUND_end.y
	// HAND_STATE : HAND_CANDIDATE Detection Success(1) or Fail(-1) 

	int i = 0, j = 0;
	int region_count=0;
	// Initialization
	BOUNDBOX[0] = w-1;	BOUNDBOX[1] = h-1;	BOUNDBOX[2] = 0;	BOUNDBOX[3] = 0;

	// If we found hand candidate
	if(HAND_STATE==1){
		for (i = 0; i < h; i++){
			for (j = 0; j < w; j++){
				if (hand[i*w + j] == (unsigned char)255){
					region_count++;
					if (BOUNDBOX[0] >= j)
						BOUNDBOX[0] = j;
					if (BOUNDBOX[1] >= i)
						BOUNDBOX[1] = i;
					if (BOUNDBOX[2] <= j)
						BOUNDBOX[2] = j;
					if (BOUNDBOX[3] <= i)
						BOUNDBOX[3] = i;
				}
			}
		}
	}

	region[0]=region_count;
	//printf("region_count : %d ,%d\n\n\n", region_count, region[0]);
};

void Bin_draw_horizontal_line(unsigned char *src, int ch_num, int width, int x1, int y1, int x2, int y2)
{
	int i = 0, j = 0;
	if (ch_num != 1)
	{
		for (i = x1; i < x2; i++)
		{
			src[0 + 3 * i + y1*width * 3] = (unsigned char)0;
			src[1 + 3 * i + y1*width * 3] = (unsigned char)0;
			src[2 + 3 * i + y1*width * 3] = (unsigned char)255;
		}
	}
	else
	{
		for (i = x1; i < x2; i++)
		{
			src[0 + i + y1*width] = (unsigned char)0;
			src[1 + i + y1*width] = (unsigned char)0;
			src[2 + i + y1*width] = (unsigned char)255;
		}
	}

};
void Bin_draw_vertical_line(unsigned char *src, int ch_num, int width, int x1, int y1, int x2, int y2)
{
	int i = 0, j = 0;
	if (ch_num != 1)
	{
		for (i = y1; i < y2; i++)
		{
			src[0 + 3 * x1 + i*width * 3] = (unsigned char)0;
			src[1 + 3 * x1 + i*width * 3] = (unsigned char)0;
			src[2 + 3 * x1 + i*width * 3] = (unsigned char)255;
		}
	}
	else
	{
		for (i = y1; i < y2; i++)
		{
			src[0 + x1 + i*width] = (unsigned char)0;
			src[1 + x1 + i*width] = (unsigned char)0;
			src[2 + x1 + i*width] = (unsigned char)255;
		}
	}
};
void Bin_draw_Rectangle(unsigned char *src, int ch_num, int img_width, int x1, int y1, int x2, int y2)
{
	if((x1&&x2&&y1&&y2)>=0)
	{
		if (ch_num == 3)
		{
			Bin_draw_horizontal_line(src, 3, img_width, x1, y1, x2, y1);
			Bin_draw_horizontal_line(src, 3, img_width, x1, y2, x2, y2);
			Bin_draw_vertical_line(src, 3, img_width, x1, y1, x1, y2);
			Bin_draw_vertical_line(src, 3, img_width, x2, y1, x2, y2);
		}
		else
		{
			Bin_draw_horizontal_line(src, 1, img_width, x1, y1, x2, y1);
			Bin_draw_horizontal_line(src, 1, img_width, x1, y2, x2, y2);
			Bin_draw_vertical_line(src, 1, img_width, x1, y1, x1, y2);
			Bin_draw_vertical_line(src, 1, img_width, x2, y1, x2, y2);
		}
	}
};

void Bin_HAND_VERIFY(unsigned char *hand, const int w, const int h, int *BOUNDBOX, int *HAND_class, const int HAND_STATE)
{
	// This function has 3 steps(I, II)
	// 	I.   Bounding box thresholding
	//	II.  Half length Thresholding  and  Hole/Hand ratio thresholding
	//
	// hand : Hand candidate image
	// w : width
	// h : height
	// BOUNDBOX = Bounding box(hand candidate)
	// HAND_class[0] == 1 : Bounding box thresholding success     /  -1 : Bounding box thresholding fail
	// HAND_class[1] == 1 : Hole/Hand ratio thresholding success  /  -1 : Hole/Hand ratio thresholding fail 
	int i = 0, j = 0;
	int count_hole = 0, count_half = 0, num_hole = 0, num_hand = 0;
	int min_x = BOUNDBOX[2], max_x = BOUNDBOX[0];

	memset(HAND_class, 0, sizeof(int)*2);

	if(HAND_STATE == 1)
	{
		// Step I. Bounding box thresholding
		if( ((BOUNDBOX[2]-BOUNDBOX[0]) >= (w*8/10)) 
			&& ((BOUNDBOX[3]-BOUNDBOX[1]) >= (h*8/10)) 
			&& ((BOUNDBOX[2]-BOUNDBOX[0]) <= (w*1/10)) 
			&& ((BOUNDBOX[3]-BOUNDBOX[1]) <= (h*1/10))
			)
		{
			HAND_class[0] = -1;
			HAND_class[1] = -1;
		}
		else
		{
			HAND_class[0]=1;
			HAND_class[1]=1;
		}


	}
};

void Bin_Pointing_Vector(unsigned char *hand, int w, int h, int orientation, int *BOUNDBOX, int *TIP, int *BASE, int *frame_storage, int distance_threshold)
{
	// This function has 3 steps(I, II, III)
	// 	I.   Calculate the hand center
	//	II.  Generate the distance image & Fingertip detection
	//	III. Finger base detection
	//
	// hand : hand region image ( after wrist or lower palm removal process )
	// w : width
	// h : height
	// finger_type = 1, 2, 3, 4
	//     1. upper side 	2. lower side	 ,
	//	   3. left side  	4. right side
	// BOUNDBOX : Hand bounding box
	// TIP : TIP[0]  = FINGERTIP (X axis value),  TIP[1] =  FINGERTIP (Y axis value)
	// BASE: BASE[0] = FINGERBASE(X axis value),  BASE[1] = FINGERBASE(Y axis value)
	// Frame_storage: Frame_storage[0] = # of hand detection success, Frame_storage[1] = frame storage
	int i = 0, j = 0, count = 0;

	int HAND_CENTER[2] = { 0 };
	int TIP_prev[2] = { 0 }, BASE_prev[2] = { 0 }, BASE_temp[2] = { 0 };
	int thresh_tracking = 30;
	int thresh_frame = 0;
	int dist_max = 0;
	int dist = 0;

	int center_min = 0, center_max = 0;

	int THRESH_FINGER = 0;
	int ratio = 30;


	TIP_prev[0] = TIP[0];
	TIP_prev[1] = TIP[1];
	BASE_prev[0] = BASE[0];
	BASE_prev[1] = BASE[1];

	memset(TIP, 0, sizeof(int)*2);
	memset(BASE,0, sizeof(int)*2);

	if (orientation == 1)
	{
		// I. Calculate the hand center

		center_min = BOUNDBOX[3];
		center_max = BOUNDBOX[1];

		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			if (hand[i*w + ((BOUNDBOX[0] + BOUNDBOX[2])/2 -1)] != (unsigned char)0)
			{
				center_min = (center_min > i) ? i : center_min;
				center_max = (center_max < i) ? i : center_max;
			}
		}
		HAND_CENTER[0] = (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2) - 1);
		HAND_CENTER[1] = (center_min + center_max) / 2;

		// II. Generate the distance map & Fingertip detection
		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j <= (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2) - 1); j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					if (dist_max < dist)
					{
						dist_max = dist;
						// FINGERTIP detection
						TIP[0] = j;
						TIP[1] = i;
					}
				}
			}
		}
		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j <= (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2) - 1); j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					hand[i*w + j] = (unsigned char)(dist * 255 / dist_max);
				}
			}
		}

		// III. Finger base detection
		THRESH_FINGER = dist_max / ratio;

		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j <= (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2) - 1); j++)
			{
				if (TIP[1] < (BOUNDBOX[1] + BOUNDBOX[3]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}

				else if (TIP[1] >= (BOUNDBOX[1] + BOUNDBOX[3]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}
			}
		}
		if ((frame_storage[0] > 0) && (frame_storage[1] < thresh_frame)
			&& (((TIP[0] - TIP_prev[0] > thresh_tracking) || (TIP[0] - TIP_prev[0] < -thresh_tracking))
			|| ((TIP[1] - TIP_prev[1] > thresh_tracking) || (TIP[1] - TIP_prev[1] < -thresh_tracking))))
		{
			TIP[0] = TIP_prev[0];
			TIP[1] = TIP_prev[1];
			BASE[0] = BASE_prev[0];
			BASE[1] = BASE_prev[1];
			frame_storage[1] = frame_storage[1] + 1;
		}
		else if(count!=0)
		{
			BASE[0] = (int)(BASE[0] / count);
			BASE[1] = (int)(BASE[1] / count);
			frame_storage[1] = 0;
		}
		else 
			return ;
		frame_storage[0] = frame_storage[0] + 1;
		count = 0;
	}


	else if (orientation == 2)
	{
		// I. Calculate the hand center
		center_min = BOUNDBOX[3];
		center_max = BOUNDBOX[1];

		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			if (hand[i*w + ((BOUNDBOX[0] + BOUNDBOX[2]) / 2)] != (unsigned char)0)
			{
				center_min = (center_min > i) ? i : center_min;
				center_max = (center_max < i) ? i : center_max;
			}
		}
		HAND_CENTER[0] = BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2);
		HAND_CENTER[1] = (center_min + center_max) / 2;

		// II. Generate the distance map & Fingertip detection
		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2)); j <= BOUNDBOX[2]; j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					if (dist_max < dist)
					{
						dist_max = dist;
						TIP[0] = j;
						TIP[1] = i;
					}
				}
			}
		}
		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2)); j <= BOUNDBOX[2]; j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					hand[i*w + j] = (unsigned char)(dist * 255 / dist_max);
				}
			}
		}

		// III. Finger base detection
		THRESH_FINGER = dist_max / ratio;

		for (i = BOUNDBOX[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = (BOUNDBOX[0] + ((BOUNDBOX[2] - BOUNDBOX[0]) / 2)); j <= BOUNDBOX[2]; j++)
			{
				if (TIP[1] < (BOUNDBOX[1] + BOUNDBOX[3]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}

				else if (TIP[1] >= (BOUNDBOX[1] + BOUNDBOX[3]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}
			}
		}
		if ((frame_storage[0] > 0) && (frame_storage[1] < thresh_frame)
			&& (((TIP[0] - TIP_prev[0] > thresh_tracking) || (TIP[0] - TIP_prev[0] < -thresh_tracking))
			|| ((TIP[1] - TIP_prev[1] > thresh_tracking) || (TIP[1] - TIP_prev[1] < -thresh_tracking))))
		{
			TIP[0] = TIP_prev[0];
			TIP[1] = TIP_prev[1];
			BASE[0] = BASE_prev[0];
			BASE[1] = BASE_prev[1];
			frame_storage[1] = frame_storage[1] + 1;
		}
		else if(count!=0)
		{
			BASE[0] = (int)(BASE[0] / count);
			BASE[1] = (int)(BASE[1] / count);
			frame_storage[1] = 0;
		}
		else 
			return ;
		frame_storage[0] = frame_storage[0] + 1;
		count = 0;
	}

	else if (orientation == 3)
	{
		// I. Calculate the hand center

		center_min = BOUNDBOX[2];
		center_max = BOUNDBOX[0];

		for (i = BOUNDBOX[0]; i <= BOUNDBOX[2]; i++)
		{
			if (hand[(((BOUNDBOX[1] + BOUNDBOX[3]) / 2) - 1)*w + i] != (unsigned char)0)
			{
				center_min = (center_min > i) ? i : center_min;
				center_max = (center_max < i) ? i : center_max;
			}
		}
		HAND_CENTER[0] = (center_min + center_max) / 2;
		HAND_CENTER[1] = BOUNDBOX[1] + ((BOUNDBOX[3] - BOUNDBOX[1]) / 2) - 1;

		// II. Generate the distance map & Fingertip detection
		for (i = BOUNDBOX[1]; i <= HAND_CENTER[1]; i++)
		{
			for (j = BOUNDBOX[0]; j < BOUNDBOX[2]; j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					if (dist_max < dist)
					{
						dist_max = dist;
						// FINGERTIP detection
						TIP[0] = j;
						TIP[1] = i;
					}

				}
			}
		}

		for (i = BOUNDBOX[1]; i <= HAND_CENTER[1]; i++)
		{
			for (j = BOUNDBOX[0]; j <= BOUNDBOX[2]; j++)
			{
				if ((hand[i*w + j] != (unsigned char)0) && dist_max!=0 )
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					hand[i*w + j] = (unsigned char)(dist * 255 / dist_max);
				}

			}
		}

		//III. Finger base detection
		THRESH_FINGER = dist_max / ratio;

		for (i = BOUNDBOX[1]; i <= HAND_CENTER[1]; i++)
		{
			for (j = BOUNDBOX[0]; j <= BOUNDBOX[2]; j++)
			{
				if (TIP[0] < (BOUNDBOX[0] + BOUNDBOX[2]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}

				else if (TIP[0] >= (BOUNDBOX[0] + BOUNDBOX[2]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}
			}
		}

		if ((frame_storage[0] > 0) && (frame_storage[1] < thresh_frame)
			&& (((TIP[0] - TIP_prev[0] > thresh_tracking) || (TIP[0] - TIP_prev[0] < -thresh_tracking))
			|| ((TIP[1] - TIP_prev[1] > thresh_tracking) || (TIP[1] - TIP_prev[1] < -thresh_tracking))))
		{
			TIP[0] = TIP_prev[0];
			TIP[1] = TIP_prev[1];
			BASE[0] = BASE_prev[0];
			BASE[1] = BASE_prev[1];
			frame_storage[1] = frame_storage[1] + 1;
		}
		else if(count!=0)
		{
			BASE[0] = (int)(BASE[0] / count);
			BASE[1] = (int)(BASE[1] / count);
			frame_storage[1] = 0;
		}
		else 
			return ;

		frame_storage[0] = frame_storage[0] + 1;
		count = 0;

	}

	else if (orientation == 4)
	{
		// I. Calculate the hand center
		center_min = BOUNDBOX[2];
		center_max = BOUNDBOX[0];

		for (i = BOUNDBOX[0]; i <= BOUNDBOX[2]; i++)
		{
			if (hand[((BOUNDBOX[1] + BOUNDBOX[3]) / 2)*w + i] != (unsigned char)0)
			{
				center_min = (center_min > i) ? i : center_min;
				center_max = (center_max < i) ? i : center_max;
			}
		}
		HAND_CENTER[0] = (center_min + center_max) / 2;
		HAND_CENTER[1] = BOUNDBOX[1] + ((BOUNDBOX[3] - BOUNDBOX[1]) / 2);

		// II. Generate the distance map & Fingertip detection
		for (i = HAND_CENTER[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j < BOUNDBOX[2]; j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					if (dist_max < dist)
					{
						dist_max = dist;
						// FINGERTIP detection
						TIP[0] = j;
						TIP[1] = i;
					}
				}
			}
		}
		for (i = HAND_CENTER[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j <= BOUNDBOX[2]; j++)
			{
				if (hand[i*w + j] != (unsigned char)0)
				{
					dist = (j - HAND_CENTER[0])*(j - HAND_CENTER[0]) + (i - HAND_CENTER[1]) * (i - HAND_CENTER[1]);
					hand[i*w + j] = (unsigned char)(dist * 255 / dist_max);
				}
			}
		}

		// III. Finger base detection
		THRESH_FINGER = dist_max / ratio;

		for (i = HAND_CENTER[1]; i <= BOUNDBOX[3]; i++)
		{
			for (j = BOUNDBOX[0]; j <= BOUNDBOX[2]; j++)
			{
				if (TIP[0] < (BOUNDBOX[0] + BOUNDBOX[2]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}

				else if (TIP[0] >= (BOUNDBOX[0] + BOUNDBOX[2]) / 2)
				{
					if ((hand[i*w + j] != (unsigned char)0)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) < THRESH_FINGER)
						&& ((TIP[0] - j)*(TIP[0] - j) + (TIP[1] - i)*(TIP[1] - i) > THRESH_FINGER / 10))
					{
						BASE[0] = BASE[0] + j;
						BASE[1] = BASE[1] + i;
						count = count + 1;
					}
				}
			}
		}
		if ((frame_storage[0] > 0) && (frame_storage[1] < thresh_frame)
			&& (((TIP[0] - TIP_prev[0] > thresh_tracking) || (TIP[0] - TIP_prev[0] < -thresh_tracking))
			|| ((TIP[1] - TIP_prev[1] > thresh_tracking) || (TIP[1] - TIP_prev[1] < -thresh_tracking))))
		{
			TIP[0] = TIP_prev[0];
			TIP[1] = TIP_prev[1];
			BASE[0] = BASE_prev[0];
			BASE[1] = BASE_prev[1];
			frame_storage[1] = frame_storage[1] + 1;
		}
		else if(count!=0)
		{
			BASE[0] = (int)(BASE[0] / count);
			BASE[1] = (int)(BASE[1] / count);
			frame_storage[1] = 0;
		}
		else 
			return ;
		frame_storage[0] = frame_storage[0] + 1;
		count = 0;
	}
};

void Bin_Color_Extraction(unsigned char *color_3ch, int img_width, int x1, int y1, int x2, int y2, int ROI_type, int *color_palette)
{
	// This function has 3 steps(I, II)
	// 	I.   Generate the ROI corresponding to ROI_type
	//	II.  Color analysis using HSV histogram 
	//
	// color_3ch: Finger Segmentation image ( BGR color image )
	// img_width: width of color_3ch
	// x1: ROI left upper point(X axis value)
	// y1: ROI left upper point(Y axis value)
	// x2: ROI right lower point(X axis value)
	// y2: ROI right lower point(Y axis value)
	// ROI_type: 1 == Rectangle, 2 == Enclosed circle in Rectangle, 3 == Half-Circle
	// color_palette[0] = 1st color, [1] = 2nd color, [2] = 3rd color


	// Color Sequence :  Red -> Yellow-> Green -> Blue -> White -> Black
	// Color Labeling :   1  ->   2   ->   3   ->  4   ->   5   ->   6  
	int *color_histogram = (int*)calloc(6, sizeof(int)); // Color Histogram
	int *color_label = (int*)calloc(6, sizeof(int)); // Color labeling

	int i = x1, j = y1;
	int R = 0, G = 0, B = 0;

	// Color Histogram Variables
	int maximum_value = 0, minimum_value = 0, computed_H = 0, computed_S = 0, computed_V = 0;

	int r_0 = 0;
	int radius_x = 0, radius_y = 0;
	int r_1 = 0;

	for (i = 0; i < 6; i++)
		color_label[i] = (i + 1);

	// Initialization color palette
	memset(color_palette, 0, sizeof(int)*3);

	// Rectangle ROI
	if (ROI_type == 0)
	{

		// Generate the Color Histogram ( RGB, HSV )
		for (i = x1; i < x2; i++)
		{   
			for (j = y1; j < y2; j++)
			{
				// RGB 2 HSV
				B = ((int)(uchar)color_3ch[0 + 3 * i + (j*img_width * 3)] * 100) / 255;
				G = ((int)(uchar)color_3ch[1 + 3 * i + (j*img_width * 3)] * 100) / 255;
				R = ((int)(uchar)color_3ch[2 + 3 * i + (j*img_width * 3)] * 100) / 255;

				color_3ch[0 + 3 * i + (j*img_width * 3)] = 0;
				color_3ch[1 + 3 * i + (j*img_width * 3)] = 0;
				color_3ch[2 + 3 * i + (j*img_width * 3)] = 255;

				maximum_value = B <= G ? G : B;
				maximum_value = maximum_value <= R ? R : maximum_value;
				minimum_value = B <= G ? B : G;
				minimum_value = minimum_value <= R ? minimum_value : R;

				if (minimum_value == maximum_value)
				{
					computed_H = 0; computed_S = 0, computed_V = minimum_value;
				}
				else
				{
					int d = (R == minimum_value) ? (G - B) : ((B == minimum_value) ? R - G : B - R);
					int h = (R == minimum_value) ? 180 : ((B == minimum_value) ? 60 : 300);
					computed_H = h - ((d * 60) / (maximum_value - minimum_value)); //H : 0~360degree
					computed_S = (100 * (maximum_value - minimum_value)) / maximum_value; //S : 0~100
					computed_V = maximum_value; // V: 0~100
					//printf("%d, %d, %d,\n", computed_H, computed_S, computed_V);
				}

				// Red
				if ((computed_H<30 || computed_H>310) && (computed_S>50) && (computed_V>50))
					color_histogram[0] += 1;
				// Yellow
				else if (computed_H >= 50 && computed_H<70 && (computed_S>50) && (computed_V>50))
					color_histogram[1] += 1;
				// Green
				else if (computed_H >= 100 && computed_H<140 && (computed_S>50) && (computed_V>50))
					color_histogram[2] += 1;
				// Blue
				else if (computed_H >= 200 && computed_H<240 && (computed_S>50) && (computed_V>50))
					color_histogram[3] += 1;
				// Black
				else if (computed_V < 30)
					color_histogram[5] += 1;
				// White
				else if (computed_S<30 && computed_V>60)
					color_histogram[4] += 1;
			}
		}
	}


	// Enclose Circle ROI
	if (ROI_type == 1)
	{
		// Enclose circle r_0
		r_0 = (x2 - x1) <= (y2 - y1) ? (x2 - x1) / 2 : (y2 - y1) / 2;
		radius_x = (x2 + x1) / 2;
		radius_y = (y2 + y1) / 2;

		// Generate the Color Histogram ( RGB, HSV )
		for (i = x1; i < x2; i++)
		{
			for (j = y1; j < y2; j++)
			{
				if (r_0*r_0 > (i - radius_x)*(i - radius_x) + (j - radius_y)*(j - radius_y))
				{
					// RGB 2 HSV
					B = ((int)(uchar)color_3ch[0 + 3 * i + (j*img_width * 3)] * 100) / 255;
					G = ((int)(uchar)color_3ch[1 + 3 * i + (j*img_width * 3)] * 100) / 255;
					R = ((int)(uchar)color_3ch[2 + 3 * i + (j*img_width * 3)] * 100) / 255;

					color_3ch[0 + 3 * i + (j*img_width * 3)] = 0;
					color_3ch[1 + 3 * i + (j*img_width * 3)] = 0;
					color_3ch[2 + 3 * i + (j*img_width * 3)] = 255;

					maximum_value = B <= G ? G : B;
					maximum_value = maximum_value <= R ? R : maximum_value;
					minimum_value = B <= G ? B : G;
					minimum_value = minimum_value <= R ? minimum_value : R;

					if (minimum_value == maximum_value)
					{
						computed_H = 0; computed_S = 0, computed_V = minimum_value;
					}
					else
					{
						int d = (R == minimum_value) ? (G - B) : ((B == minimum_value) ? R - G : B - R);
						int h = (R == minimum_value) ? 180 : ((B == minimum_value) ? 60 : 300);
						computed_H = h - ((d * 60) / (maximum_value - minimum_value)); //H : 0~360degree
						computed_S = (100 * (maximum_value - minimum_value)) / maximum_value; //S : 0~100
						computed_V = maximum_value; // V: 0~100
						//printf("%d, %d, %d,\n", computed_H, computed_S, computed_V);
					}

					// Red
					if ((computed_H<30 || computed_H>310) && (computed_S>50) && (computed_V>50))
						color_histogram[0] += 1;
					// Yellow
					else if (computed_H >= 50 && computed_H<70 && (computed_S>50) && (computed_V>50))
						color_histogram[1] += 1;
					// Green
					else if (computed_H >= 100 && computed_H<140 && (computed_S>50) && (computed_V>50))
						color_histogram[2] += 1;
					// Blue
					else if (computed_H >= 200 && computed_H<240 && (computed_S>50) && (computed_V>50))
						color_histogram[3] += 1;
					// Black
					else if (computed_V < 30)
						color_histogram[5] += 1;
					// White
					else if (computed_S<30 && computed_V>60)
						color_histogram[4] += 1;
				}

			}
		}
	}

	// Half-Circle ROI
	if (ROI_type == 2)
	{
		int radius_x = (x2 + x1) / 2;
		int radius_y = y2;

		r_1 = y2 - y1;	// Half-circle r_1

		// Generate the Color Histogram ( RGB, HSV )
		for (i = x1; i < x2; i++)
		{
			for (j = y1; j < y2; j++)
			{
				if (r_1*r_1 >(i - radius_x)*(i - radius_x) + (j - radius_y)*(j - radius_y))
				{
					// RGB 2 HSV
					B = ((int)(uchar)color_3ch[0 + 3 * i + (j*img_width * 3)] * 100) / 255;
					G = ((int)(uchar)color_3ch[1 + 3 * i + (j*img_width * 3)] * 100) / 255;
					R = ((int)(uchar)color_3ch[2 + 3 * i + (j*img_width * 3)] * 100) / 255;

					color_3ch[0 + 3 * i + (j*img_width * 3)] = 0;
					color_3ch[1 + 3 * i + (j*img_width * 3)] = 0;
					color_3ch[2 + 3 * i + (j*img_width * 3)] = 255;

					maximum_value = B <= G ? G : B;
					maximum_value = maximum_value <= R ? R : maximum_value;
					minimum_value = B <= G ? B : G;
					minimum_value = minimum_value <= R ? minimum_value : R;

					if (minimum_value == maximum_value)
					{
						computed_H = 0; computed_S = 0, computed_V = minimum_value;
					}
					else
					{
						int d = (R == minimum_value) ? (G - B) : ((B == minimum_value) ? R - G : B - R);
						int h = (R == minimum_value) ? 180 : ((B == minimum_value) ? 60 : 300);
						computed_H = h - ((d * 60) / (maximum_value - minimum_value)); //H : 0~360degree
						computed_S = (100 * (maximum_value - minimum_value)) / maximum_value; //S : 0~100
						computed_V = maximum_value; // V: 0~100
						//printf("%d, %d, %d,\n", computed_H, computed_S, computed_V);
					}

					// Red
					if ((computed_H<30 || computed_H>310) && (computed_S>50) && (computed_V>50))
						color_histogram[0] += 1;
					// Yellow
					else if (computed_H >= 50 && computed_H<70 && (computed_S>50) && (computed_V>50))
						color_histogram[1] += 1;
					// Green
					else if (computed_H >= 100 && computed_H<140 && (computed_S>50) && (computed_V>50))
						color_histogram[2] += 1;
					// Blue
					else if (computed_H >= 200 && computed_H<240 && (computed_S>50) && (computed_V>50))
						color_histogram[3] += 1;
					// Black
					else if (computed_V < 30)
						color_histogram[5] += 1;
					// White
					else if (computed_S<30 && computed_V>60)
						color_histogram[4] += 1;
				}

			}
		}
	}

	//maximum_value=color_histogram[0]<color_histogram[1]?color_histogram[1]:color_histogram[0];
	//maximum_value=maximum_value<color_histogram[2]?color_histogram[2]:maximum_value;
	//maximum_value=maximum_value<color_histogram[3]?color_histogram[3]:maximum_value;
	//maximum_value=maximum_value<color_histogram[4]?color_histogram[4]:maximum_value;
	//maximum_value=maximum_value<color_histogram[5]?color_histogram[5]:maximum_value;
	maximum_value = 0;
	for (i = 0; i < 6; i++)
		maximum_value += color_histogram[i];
	if (maximum_value != 0)
		for (i = 0; i < 6; i++)
			color_histogram[i] = (100 * color_histogram[i]) / maximum_value;

	printf("Red : %d, Yellow : %d, Green : %d, Blue : %d\nWhite : %d, Black : %d, \n", color_histogram[0], color_histogram[1],
		color_histogram[2], color_histogram[3], color_histogram[4], color_histogram[5]);


	Bin_sorting_2D(color_histogram, color_label, 6);
	if (color_histogram[5] >= 5)
	{
		printf("1st Color : ");
		switch (color_label[5])
		{
		case 1: printf("Red\n"); color_palette[0] = 1; break;
		case 2: printf("Yellow\n"); color_palette[0] = 2; break;
		case 3: printf("Green\n"); color_palette[0] = 3; break;
		case 4: printf("Blue\n"); color_palette[0] = 4; break;
		case 5: printf("White\n"); color_palette[0] = 5; break;
		case 6: printf("Black\n"); color_palette[0] = 6; break;
		}
		if (color_histogram[4] >= 5)
		{
			printf("2nd Color : ");
			switch (color_label[4])
			{
			case 1: printf("Red\n"); color_palette[1] = 1; break;
			case 2: printf("Yellow\n"); color_palette[1] = 2; break;
			case 3: printf("Green\n"); color_palette[1] = 3; break;
			case 4: printf("Blue\n"); color_palette[1] = 4; break;
			case 5: printf("White\n"); color_palette[1] = 5; break;
			case 6: printf("Black\n"); color_palette[1] = 6; break;
			}
			if (color_histogram[3] >= 5)
			{
				printf("3rd Color : ");
				switch (color_label[3])
				{
				case 1: printf("Red\n"); color_palette[2] = 1; break;
				case 2: printf("Yellow\n"); color_palette[2] = 2; break;
				case 3: printf("Green\n"); color_palette[2] = 3; break;
				case 4: printf("Blue\n"); color_palette[2] = 4; break;
				case 5: printf("White\n"); color_palette[2] = 5; break;
				case 6: printf("Black\n"); color_palette[2] = 6; break;
				}
			}
		}
		printf("\n");
	}
	else
		printf("This Color is not defined.\n\n");

	free(color_histogram);
	free(color_label);
};






void Bin_A255B255C255toD255(unsigned char *A, unsigned char *B, unsigned char *C, unsigned char *D, int length)
{
	int i=0;
	for( i = 0 ; i < length ; i++)
	{
		if((int)A[i]!=0||
			(int)B[i]!=0||
			(int)C[i]!=0
			)
			D[i]=(unsigned char)255;
		else
			D[i]=(unsigned char)0;

	}


};

void Bin_A255B255toD255(unsigned char *A, unsigned char *B, unsigned char *D, int length)
{
	int i=0;
	for( i = 0 ; i < length ; i++)
	{
		if((int)A[i]!=0||
			(int)B[i]!=0
			)
			D[i]=(unsigned char)255;
		else
			D[i]=(unsigned char)0;

	}

};


void Bin_IMG_rotation(unsigned char *src, unsigned char *dst, int nchannel, int width_src, int height_src, int width_dst, int height_dst)
{
	int i =0, j=0;
	int w = width_src;
	int h = height_src;
	int nch = nchannel;
	unsigned char *temp0 = (unsigned char*)calloc(w*h,sizeof(unsigned char));
	unsigned char *temp1 = (unsigned char*)calloc(w*h,sizeof(unsigned char));
	unsigned char *temp2 = (unsigned char*)calloc(w*h,sizeof(unsigned char));
	unsigned char *temp3 = (unsigned char*)calloc(w*h,sizeof(unsigned char));
	int w_dst = width_dst;
	int h_dst = height_dst;

	int count=0;
	for(j = 0 ; j <h ; j++)
	{
		for(i = 0 ; i < w ; i++)
		{

			if(nch==1)	
			{
				temp0[count]=src[i+j*w];				count++;
			}
			else if(nch==3)	
			{
				temp0[count]=src[i*3+0+j*3*w];				
				temp1[count]=src[i*3+1+j*3*w];				
				temp2[count]=src[i*3+2+j*3*w];				count++;

			}
			else if(nch==4)
			{
				temp0[count]=src[i*4+0+j*4*w];				
				temp1[count]=src[i*4+1+j*4*w];				
				temp2[count]=src[i*4+2+j*4*w];				
				temp3[count]=src[i*4+3+j*4*w];				count++;
			}
		}
	}
	count=0;
	for(j = 0 ; j <w_dst ; j++)
	{
		for(i = h_dst-1 ; i >= 0 ; i--)
		{
			if(nch==1)	
			{
				dst[j+i*w_dst]=temp0[count];				
				count++;
			}

			else if(nch==3)	
			{
				dst[j*3+0+i*w_dst]=temp0[count];
				dst[j*3+1+i*w_dst]=temp1[count];
				dst[j*3+2+i*w_dst]=temp2[count];
				count++;
			}
			else if(nch==4)
			{
				dst[j*4+0+i*4*w_dst]=temp0[count];
				dst[j*4+1+i*4*w_dst]=temp1[count];
				dst[j*4+2+i*4*w_dst]=temp2[count];
				dst[j*4+3+i*4*w_dst]=temp3[count];
				count++;

			}
		}
	}




	free(temp0);
	free(temp1);
	free(temp2);
	free(temp3);


};


void Bin_ROI_SEED(unsigned char *hand, int w, int h, int finger_type, int *BOUNDBOX, int *TIP, int *BASE, int *SEED, int *ROI, const int ROI_length)
{
	// This function has 3 steps(I, II)
	// 	I.   Calculate ROI SEED
	//	II.  Extract the start and end point of the ROI
	//
	// hand : Finger Segmentation image
	// w : width
	// h : height
	// TIP : TIP[0]  = FINGERTIP (X axis value),  TIP[1] =  FINGERTIP (Y axis value)
	// BASE: BASE[0] = FINGERBASE(X axis value),  BASE[1] = FINGERBASE(Y axis value)
	// SEED: SEED[0] = ROI SEED(X axis value),  SEED[1] = ROI SEED(Y axis value)
	// ROI[0] =  ROI left upper point(X axis value),  ROI[1] = ROI left upper point(Y axis value)
	// ROI[2] =  ROI right lower point(X axis value),  ROI[3] = ROI right lower point(Y axis value)

	int i = 0, j = 0;

	int m = 2, n = 1;	// ratio setting m:n ( ROI SEED )

	// I. Calculate the point of external division ( ROI SEED )
	SEED[0] = (m*TIP[0] - n*BASE[0]) / (m - n);
	SEED[1] = (m*TIP[1] - n*BASE[1]) / (m - n);

	//	II.  Extract the start and end point of the ROI
	if(finger_type==1){
		ROI[0] = SEED[0] - ROI_length;
		ROI[1] = SEED[1] - ROI_length;
		ROI[2] = SEED[0] + ROI_length;
		ROI[3] = SEED[1];

		if(ROI[3] > BOUNDBOX[1]){
			ROI[2] = SEED[0];
			ROI[3] = SEED[1] + ROI_length;

			if(ROI[2] > BOUNDBOX[0]){
				ROI[0] = SEED[0] - ROI_length;
				ROI[1] = SEED[1] - ROI_length;
				ROI[2] = SEED[0] + ROI_length;
				ROI[3] = SEED[1];
			}
		}
	}	
	else if(finger_type==2){
		ROI[0] = SEED[0] - ROI_length;
		ROI[1] = SEED[1];
		ROI[2] = SEED[0] + ROI_length;
		ROI[3] = SEED[1] + ROI_length;

		if(ROI[1] < BOUNDBOX[3]){
			ROI[0] = SEED[0];
			ROI[1] = SEED[1] - ROI_length;

			if(ROI[0] < BOUNDBOX[2]){
				ROI[0] = SEED[0] - ROI_length;
				ROI[1] = SEED[1];
				ROI[2] = SEED[0] + ROI_length;
				ROI[3] = SEED[1] + ROI_length;
			}
		}
	}
	else if(finger_type==3){
		ROI[0] = SEED[0] - ROI_length;
		ROI[1] = SEED[1] - ROI_length;
		ROI[2] = SEED[0];
		ROI[3] = SEED[1] + ROI_length;

		if(ROI[2] > BOUNDBOX[0]){
			ROI[2] = SEED[0] + ROI_length;
			ROI[3] = SEED[1];

			if(ROI[3] > BOUNDBOX[1]){
				ROI[0] = SEED[0] - ROI_length;
				ROI[1] = SEED[1] - ROI_length;
				ROI[2] = SEED[0];
				ROI[3] = SEED[1] + ROI_length;
			}
		}
	}
	else if(finger_type==4){
		ROI[0] = SEED[0];
		ROI[1] = SEED[1] - ROI_length;
		ROI[2] = SEED[0] + ROI_length;
		ROI[3] = SEED[1] + ROI_length;

		if(ROI[0] < BOUNDBOX[2]){
			ROI[0] = SEED[0] - ROI_length;
			ROI[1] = SEED[1];

			if(ROI[1] < BOUNDBOX[3]){
				ROI[0] = SEED[0];
				ROI[1] = SEED[1] - ROI_length;
				ROI[2] = SEED[0] + ROI_length;
				ROI[3] = SEED[1] + ROI_length;
			}
		}
	}

	// Exception handling
	if (ROI[0] < 0)	ROI[0] = 0;
	else if(ROI[0] > w)	ROI[0] = w;

	if (ROI[1] < 0)	ROI[1] = 0;
	else if(ROI[1] > h)	ROI[1] = h;

	if (ROI[2] < 0)	ROI[2] = 0;
	else if(ROI[2] > w)	ROI[2] = w;

	if (ROI[3] < 0)	ROI[3] = 0;
	else if(ROI[3] > h)	ROI[3] = h;
};
void Bin_ROI(unsigned char *hand, int w, int h, int *TIP, int *BASE, int *ROI, const int ROI_extension_ratio)
{
	// This function has 3 steps(I, II)
	// 	I.   Calculate ROI SEED
	//	II.  Classify the slope type of the pointing vector
	//  III. Generate the ROI ( Output : start and end point of ROI )
	//
	// hand : Finger Segmentation image
	// w : width
	// h : height
	// TIP : TIP[0]  = FINGERTIP (X axis value),  TIP[1] =  FINGERTIP (Y axis value)
	// BASE: BASE[0] = FINGERBASE(X axis value),  BASE[1] = FINGERBASE(Y axis value)
	// ROI[0] =  ROI left upper point(X axis value),  ROI[1] = ROI left upper point(Y axis value)
	// ROI[2] =  ROI right lower point(X axis value),  ROI[3] = ROI right lower point(Y axis value)
	// ROI_extension_ratio : Extract the length of ROI

	int ROI_x_near = 0, ROI_y_near = 0, ROI_x_far = 0, ROI_y_far = 0;
	int m_near = 2, n_near = 1;	// ratio setting m:n
	int m_far = 5, n_far = 3;

	int slope = 0;
	int slope_type = 0;

	int center_x = 0, center_y = 0;
	int len = 0;

	// I. Calculate the point of external division 
	ROI_x_near = (m_near*TIP[0] - n_near*BASE[0]) / (m_near - n_near);
	ROI_y_near = (m_near*TIP[1] - n_near*BASE[1]) / (m_near - n_near);

	ROI_x_far = (m_far*TIP[0] - n_far*BASE[0]) / (m_far - n_far);
	ROI_y_far = (m_far*TIP[1] - n_far*BASE[1]) / (m_far - n_far);

	//	II. Classify the slope type of the pointing vector
	if((TIP[0]-BASE[0])!=0)
		slope = (TIP[1]-BASE[1])/(TIP[0]-BASE[0]);
	else
		slope = 1;
	if( (slope >= 1 || slope <= -1) && (TIP[1] <= BASE[1]) )
		slope_type = 1;
	else if ( (slope == 0) && (TIP[0] <= BASE[0]) )
		slope_type = 2;
	else if ( (slope >= 1 || slope <= -1) && (TIP[1] > BASE[1]) )
		slope_type = 3;
	else if ( (slope == 0) && (TIP[0] > BASE[0]) )
		slope_type = 4;

	// III. Generate the ROI ( Output : start and end point of ROI )
	center_x = (ROI_x_near + ROI_x_far) / 2;
	center_y = (ROI_y_near + ROI_y_far) / 2;

	if (slope_type == 1)		
		len = ROI_y_near - ROI_y_far;
	else if(slope_type ==2)
		len = ROI_x_near - ROI_x_far;
	else if (slope_type == 3)		
		len = ROI_y_far - ROI_y_near;
	else
		len = ROI_x_far - ROI_x_near;

	len *= ROI_extension_ratio;

	// Calculate the start and end point of ROI with Exception Handling
	ROI[0] = center_x - (len/2);
	if (ROI[0] < 0)	ROI[0] = 0;
	else if(ROI[0] > w)	ROI[0] = w;

	ROI[1] = center_y - (len/2);
	if (ROI[1] < 0)	ROI[1] = 0;
	else if(ROI[1] > h)	ROI[1] = h;

	ROI[2] = center_x + (len/2);
	if ( ROI[2] < 0)	ROI[2] = 0;
	else if(ROI[2] > w)	ROI[2] = w;

	ROI[3] = center_y + (len/2);
	if (ROI[3] < 0)	ROI[3] = 0;
	else if(ROI[3] > h)	ROI[3] = h;
};
void WHITE_BALANCE(unsigned char *src_BGR, const int src_w, const int src_h, unsigned char *dst_WB)
{
	double AVG_R = 0;
	double AVG_G = 0;
	double AVG_B = 0;
	double alpha=0, beta=0;
	int i = 0, j = 0;
	for ( i = 0; i < src_h; i++)
	{
		for ( j = 0; j < src_w; j++)
		{
			AVG_R += (double)src_BGR[i*(src_w * 3) + (3 * j + 2)];
			AVG_G += (double)src_BGR[i*(src_w * 3) + (3 * j + 1)];
			AVG_B += (double)src_BGR[i*(src_w * 3) + (3 * j)];
		}
	}
	AVG_R /= (src_w*src_h);
	AVG_G /= (src_w*src_h);
	AVG_B /= (src_w*src_h);

	alpha = AVG_G / AVG_R;
	beta = AVG_G / AVG_B;

	for ( i = 0; i < src_h; i++)
	{
		for ( j = 0; j < src_w; j++)
		{
			double R_WB = alpha * (double)src_BGR[i*(src_w * 3) + (3 * j + 2)];
			double B_WB = beta * (double)src_BGR[i*(src_w * 3) + (3 * j)];

			// R channel
			if (R_WB < 0)   R_WB = 0;
			else if (R_WB > 255) R_WB = 255;
			dst_WB[i*(src_w * 3) + (3 * j + 2)] = (unsigned char)R_WB;
			// G channel
			dst_WB[i*(src_w * 3) + (3 * j + 1)] = src_BGR[i*(src_w * 3) + (3 * j + 1)];
			// B channel
			if (B_WB < 0)   B_WB = 0;
			else if (B_WB > 255) B_WB = 255;
			dst_WB[i*(src_w * 3) + (3 * j)] = (unsigned char)B_WB;

		}
	}
};

void Bin_Image_crop(unsigned char *src, int src_w, int src_h, unsigned char *dst, int dst_w, int dst_h,int x_st, int y_st, int x_en, int y_en)
{
	int i = 0 , j = 0 ;
	for(j = y_st; j < y_en ; j++)
	{
		for(i = x_st ; i <x_en ; i++)
		{
			dst[i-x_st+(j-y_st)*dst_w]=src[i+j*src_w];	
		}
	}


};


int Bin_validation_using_matching(int matching_score, unsigned char *Y, unsigned char *hand_candidate, int hand_candidate_w, int hand_candidate_h, unsigned char *patch_samples, int width_of_patch, int height_of_patch, int patch_num, int *BOUND_BOX, int device_orientation, int *left_right_hand)
{
	int i=0, j=0;
	int count=0;
	int x=0,y=0;
	int std=0;
	int mean=0;
	int num_=0;
	int patch_w = (int)(BOUND_BOX[2]-BOUND_BOX[0]);
	int patch_h = (int)(BOUND_BOX[3]-BOUND_BOX[1]);
	if(patch_w%4!=0)
	{
		if(patch_w%4==1)
		{
			BOUND_BOX[2]=BOUND_BOX[2]-1;
			patch_w= (int)(BOUND_BOX[2]-BOUND_BOX[0]);
		}
		else if(patch_w%4==2)
		{
			BOUND_BOX[2]=BOUND_BOX[2]-2;
			patch_w= (int)(BOUND_BOX[2]-BOUND_BOX[0]);
		}
		else
		{
			BOUND_BOX[2]=BOUND_BOX[2]-3;
			patch_w= (int)(BOUND_BOX[2]-BOUND_BOX[0]);
		}
	}
	if(patch_h%4!=0)
	{
		if(patch_h%4==1)
		{
			BOUND_BOX[3]=BOUND_BOX[3]-1;
			patch_h= (int)(BOUND_BOX[3]-BOUND_BOX[1]);
		}
		else if(patch_h%4==2)
		{
			BOUND_BOX[3]=BOUND_BOX[3]-2;
			patch_h= (int)(BOUND_BOX[3]-BOUND_BOX[1]);
		}
		else
		{
			BOUND_BOX[3]=BOUND_BOX[3]-3;
			patch_h= (int)(BOUND_BOX[3]-BOUND_BOX[1]);
		}
	}


	//Standarddeviation 
	for(i = BOUND_BOX[1] ; i < BOUND_BOX[3] ; i++)
	{
		for(j = BOUND_BOX[0] ; j < BOUND_BOX[2] ; j++)
		{
			if((int)hand_candidate[j+i*hand_candidate_w]!=0)
			{
				num_++;
				mean += (int)Y[j+i*hand_candidate_w];
			}
		}
	}
	mean = mean/num_;
	for(i = BOUND_BOX[1] ; i < BOUND_BOX[3] ; i++)
	{
		for(j = BOUND_BOX[0] ; j < BOUND_BOX[2] ; j++)
		{
			if((int)hand_candidate[j+i*hand_candidate_w]!=0)
			{
				std += ((int)Y[j+i*hand_candidate_w]-mean)*((int)Y[j+i*hand_candidate_w]-mean);
			}
		}
	}
	std = (int)sqrt((double)((int)std/(int)num_));
	//printf("This label has mean : %d, std : %d \n",mean, std );

	if(patch_w>=width_of_patch && patch_h>=height_of_patch && std >=9)
	{
		int threshold_distance = 75000;
		int patch_pool=0;
		int patch_x_st = BOUND_BOX[0];
		int patch_y_st = BOUND_BOX[1];
		int patch_x_en = BOUND_BOX[2];
		int patch_y_en = BOUND_BOX[3];
		int BOUND_BOX_center_x = (int)((patch_x_en+patch_x_st)/2);
		int BOUND_BOX_center_y = (int)((patch_y_en+patch_y_st)/2);
		int patch_num_st=0, patch_num_en=0;
		int patch_numnum=0;
		int nkey=0;
		int region_count=0;
		int distance_=255*width_of_patch*height_of_patch;
		unsigned char *patch = (unsigned char*)calloc(patch_w*patch_h, sizeof(unsigned char));
		unsigned char *patch_resize = (unsigned char*)calloc(width_of_patch*height_of_patch, sizeof(unsigned char));
		//IplImage *patch_sample = cvCreateImage(cvSize(patch_w, patch_h), IPL_DEPTH_8U, 1);

		Bin_Image_crop(hand_candidate,  hand_candidate_w,hand_candidate_h,patch, patch_w, patch_h, patch_x_st, patch_y_st, patch_x_en, patch_y_en);
		//SS_Array_to_Ipl2(patch, patch_sample, patch_w*patch_h);cvShowImage("patch", patch_sample);nkey=cvWaitKey(0);if(nkey=='2')cvSaveImage("./patch/000.jpg", patch_sample,0);


		if(device_orientation==2 ||device_orientation==4)
		{	
			Bin_flipud(patch, patch_w, patch_h);
			Bin_fliplr(patch, patch_w, patch_h);
		}
		Bin_ImResize_Bilinear(patch, patch_w, patch_h, width_of_patch,height_of_patch,1,patch_resize );

		for(i=0;i < width_of_patch*height_of_patch ; i++)
			if((int)patch_resize[i]!=0)
				region_count++;
		left_right_hand[0]=0;
		if(region_count < (int)(0.9*width_of_patch*height_of_patch)
			&&
			region_count > (int)(0.2*width_of_patch*height_of_patch)
			)
		{
			if(device_orientation==1 || device_orientation==2)
			{
				if(device_orientation==1)
				{
					if(BOUND_BOX_center_y<=(int)(hand_candidate_h/2))
					{
						patch_num_st=0;
						patch_num_en=(int)patch_num;
						left_right_hand[0]=1;

					}          
					else
					{
						patch_num_st=(int)patch_num;
						patch_num_en=(int)patch_num*2;
						left_right_hand[0]=-1;
					}

				}
				else if(device_orientation==2)
				{
					if(BOUND_BOX_center_y>(int)(hand_candidate_h/2))
					{
						patch_num_st=(int)0;
						patch_num_en=(int)patch_num;
						left_right_hand[0]=+1;
					}          
					else
					{
						patch_num_st=(int)patch_num;
						patch_num_en=(int)patch_num*2;
						left_right_hand[0]=-1;
					}
					//device_orientation=1;

				}


			}
			else
			{
				if(device_orientation==3)
				{
					if(BOUND_BOX_center_x>=(int)(hand_candidate_w/2))
					{
						patch_num_st=0;
						patch_num_en=(int)patch_num;
						left_right_hand[0]=1;
					}
					else
					{
						patch_num_st=(int)patch_num;
						patch_num_en=(int)patch_num*2;
						left_right_hand[0]=-1;
					}

				}
				else
				{
					if(BOUND_BOX_center_x<(int)(hand_candidate_w/2))
					{
						patch_num_st=(int)0;
						patch_num_en=(int)patch_num;
						left_right_hand[0]=1;
					}
					else
					{
						patch_num_st=(int)patch_num;
						patch_num_en=(int)patch_num*2;
						left_right_hand[0]=-1;
					}
					//device_orientation=3;

				}
			}
			for(i = patch_num_st ; i <patch_num_en ; i++ )
			{
				int SAD=0;
				unsigned char *tmp = (unsigned char*)calloc(width_of_patch*height_of_patch, sizeof(unsigned char));
				for(j = 0 ; j < width_of_patch*height_of_patch ; j++)
					tmp[j] = patch_samples[j+i*width_of_patch*height_of_patch];
				for(x= 0 ; x<height_of_patch ; x++)
				{
					for(y = 0 ; y<width_of_patch ; y++)
					{
						if( (int)patch_resize[y+x*width_of_patch]!=(int)tmp[y+x*width_of_patch])
						{
							SAD=SAD+(int)abs((int)patch_resize[y+x*width_of_patch]-(int)tmp[y+x*width_of_patch]);
							if(SAD>threshold_distance)
							{
								x=height_of_patch;
								y=width_of_patch;
							}
						}			

					}
				}
				if(distance_>=SAD)
				{
					distance_=SAD;
					patch_numnum=i;
				}
				free(tmp);
			}
			//printf("\n threshold_distance : %d with %d th patch\n", distance_, patch_numnum);
			if(distance_<threshold_distance)
			{
				//cvWaitKey(0);

				free(patch);
				free(patch_resize);
				return distance_;
			}
			else
			{
				free(patch);
				free(patch_resize);
				return INT_MAX;
			}
		}
		else
		{
			return INT_MAX;
		}

	}
	else
	{
		return INT_MAX;
	}
};
void Bin_fliplr(unsigned char *src, int w, int h)
{
	unsigned char *tmp = (unsigned char *)calloc(w*h, sizeof(unsigned char));
	int i=0,j=0;
	for(i = 0 ; i <h ; i++)
	{
		for(j = 0 ; j <w ; j++)
		{
			tmp[i*w+(w-1)-j]=src[i*w+j];
		}
	}
	for(i = 0 ; i < w*h ; i++)
		src[i]=tmp[i];
	free(tmp);

};
void Bin_flipud(unsigned char *src, int w, int h)
{
	unsigned char *tmp = (unsigned char *)calloc(w*h, sizeof(unsigned char));
	int i=0,j=0;
	for(i = 0 ; i <w ; i++)
	{
		for(j = 0 ; j <h ; j++)
		{
			tmp[(h-1-j)*w+i]=src[j*w+i];
		}
	}
	for(i = 0 ; i < w*h ; i++)
		src[i]=tmp[i];
	free(tmp);
};

void Binarize(IplImage *srcIpl, IplImage *dstIpl, int width, int height, int mode)
{
	unsigned char* src;
	unsigned char* dst;
	int i, j, k, index, indexn, indexm; 
	Bin_Ipl_to_Array(srcIpl, src, width*height);
	memcpy(dst, src, width*height);

	BinarizeChar(src, dst, width, height, mode);


	Bin_Array_to_Ipl(dst, dstIpl, width*height);
	free(src);
	free(dst);
}

void BinarizeChar(unsigned char* src, unsigned char* dst, int width, int height, int mode)
{
	int win = 16;// 32; // 2win x2win : padding: 32 but all window is 64 
	int modeIn = mode; 
	float valK = 0.1; 
	int i, j, k, index, indexn, indexm;
	int maxRN; 
	int average = averageImage( src, width,  height, win*2);
	int widthPad = width + 2 * win; 
	int heightPad = height + 2 * win; 
	unsigned char* padImage = (unsigned char*)malloc(heightPad*widthPad* sizeof(unsigned char));
	unsigned char* thr = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* thr1 = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* thr2 = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* sobel = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* kImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));


	int* padImageInt = (int*)malloc(heightPad*widthPad* sizeof(int));
	int* padImageInt2 = (int*)malloc(heightPad*widthPad* sizeof(int));

	int* padImageIntG = (int*)malloc(heightPad*widthPad* sizeof(int));
	int* padImageInt2G = (int*)malloc(heightPad*widthPad* sizeof(int));

	int* LocalMean = (int*)malloc(height*width* sizeof(int));
	int* LocalStd = (int*)malloc(height*width* sizeof(int));

	unsigned char* meanImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	float* stdRImg = (float*)malloc(height*width* sizeof(float));

	//WriteImage2Jpg(src, width, height, "inputImage.jpg");
	
	// calculate the sub image padding image: 
	MakepadImage(src, padImage, width, height, win);

	WriteImage2Jpg(padImage, widthPad, heightPad, "./processDebug/padImage.jpg");
	//imwrite("./processDebug/input.png", padImage);

	//get the average image 
	subtractImage(padImage, padImageInt, padImageInt2, widthPad, heightPad, (int)average);
	
	//integral image of normal image and square image 
	integralImage(padImageInt, padImageIntG, widthPad, heightPad);
	integralImage(padImageInt2, padImageInt2G, widthPad, heightPad);

	// local mean and std of image: 
	maxRN = LocalMeanStd(padImageIntG, padImageInt2G, LocalMean, LocalStd, width, height, win);

	int maxSobel = sobelFilter(src, dst, width, height);
	WriteImage2Jpg(dst, width, height, "./processDebug/SobelImage.jpg");

	sobelImage(dst, sobel, kImg,  width, height, 250, 100);

	WriteImage2Jpg(sobel, width, height, "./processDebug/SobelImageBin.jpg");

	CalThreshold( (unsigned char*)src, (unsigned char*)dst,
		(unsigned char*)kImg, (unsigned char*)thr, (unsigned char*)thr2,
		 (int*)LocalMean, (int*)LocalStd,  width,  height,  maxRN,
		 win, valK,  average, 
		(unsigned char*) meanImg, (float*) stdRImg);

	//CalThresholdNew  // CalThresholdBGFG
	CalThresholdBGFG2((unsigned char*)src, (unsigned char*)dst,
		(unsigned char*)kImg, (unsigned char*)thr, (unsigned char*)thr2,
		(int*)LocalMean, (int*)LocalStd, width, height, maxRN,
		win, valK, average,
		(unsigned char*)meanImg, (float*)stdRImg);

	FilterForeGround((unsigned char*)dst, (unsigned char*)src,
		width, height,
		win/2);

	signImage((unsigned char*) src, (unsigned char*)meanImg, (float*) stdRImg,
		(unsigned char*) kImg, (unsigned char*) thr,  width,  height, (unsigned char*) thr1, (unsigned char*) thr2, valK);

	//threshImg((unsigned char*) src, (unsigned char*) thr, (unsigned char*) dst, width, height);

	// calculate threshold by new method: 


	//WriteImage2Jpg(dst, width, height, "./processDebug/FinalSobelImageBin.jpg");

	
	free(thr);
	free(thr1);
	free(thr2);
	free(sobel);
	free(kImg);
	free(padImage);
	free(padImageInt);
	free(padImageInt2);
	free(padImageIntG);
	free(padImageInt2G);
	free(LocalMean);
	free(LocalStd);
	free(meanImg);
	free(stdRImg);
}


// calculate the average image. 
// calculate local mean, local std
// calculate threshold image
// calculate for each pixels
int averageImage(unsigned char* src, int width, int height, int step)
{
	int i, j, index; 
	int average =0;
	double averageF;
	index = 0;
	for (i = 0; i < height; i+=step)
	{
		for (j = 0; j < width; j+=step)
		{
			average += src[i*width + j];
			//index++; 
		}
	}
	averageF = (double)average/((width/step)*(height/step));
	return (int)(averageF+0.5f); 
}

void MinMaxImage(unsigned char* src, int width, int height, int step, int * MinMax)
{
	int i, j, index;
	int average = 0;
	double averageF;
	int minVal = 255; 
	int maxVal = 0;
	index = 0;
	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			average = src[i*width + j];
			if (minVal > average)
				minVal = average; 
			if (maxVal < average)
				maxVal = average; 
			//index++; 
		}
	}
	MinMax[0] = minVal; 
	MinMax[1] = maxVal;
}


// calculate the sub image padding image: 
void MakepadImage(unsigned char* src, unsigned char* dst, int width, int height, int win)
{	
	int i, j, k, index, indexn, indexm, id, jd;
	//int average = averageImage(src, width, height, win*2);
	int x_cord = win; 
	int y_cord = win; 
	int widthPad = width + win * 2; 
	int heightPad = height + win * 2; 
	int startPos = win*(width + 2 * win) + win; 
	int startPosS = 0; 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst[startPos+ i*widthPad + j] = src[i*width + j];
		}
	}

	startPos = win * widthPad + win; 
	startPosS = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < win; j++)
			dst[startPos + i*widthPad - j -1] = src[startPosS+ i*width + j];
	}

	startPos = win * widthPad + win +width;
	startPosS = width -1;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < win; j++)
			dst[startPos + i*widthPad +j] = src[startPosS + i*width - j];
	}


	startPosS = (win*2-1)  * widthPad ;	

	for (i = 0; i < win; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			dst[i*widthPad + j] = dst[startPosS - i*widthPad + j];
		}
	}

	startPos = (win + height)  * widthPad;
	startPosS = (win + height-1)  * widthPad;
	for (i = 0; i < win; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			dst[startPos + i*widthPad + j] = dst[startPosS - i*widthPad + j];
		}
	}	
}

// calculate the sub image padding image bigger and dividable to win
void MakepadImageWin(unsigned char* src, unsigned char* dst, int width, int height, int right, int bottom, int win, int channel)
{
	// right = if (width%win =0 right =0; else, right = win- width%win, 
	// bottom = if (height%win =0 bottom =0; else, bottom = win - height%win
	int i, j;

	int widthPad = width +right;
	int heightPad = height+bottom;
	// copy original image 
	int startPos = 0;
	int startPosS = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst[startPos + i*widthPad + j] = src[startPosS+ i*width + j];
		}
	}
	// copy right
	startPos = width;
	startPosS = width-1;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < right; j++)
			dst[startPos + i*widthPad + j] = src[startPosS + i*width - j];
	}
   
	// copy bottom
	startPos = height  * widthPad;
	startPosS =  (height - 1) * widthPad;
	for (i = 0; i < bottom; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			dst[startPos + i*widthPad + j] = dst[startPosS - i*widthPad + j];
		}
	}
}


// calculate the sub image padding image bigger and dividable to win
void MakeCropPadImageWin(unsigned char* src, unsigned char* dst, int width, int height, int right, int bottom, int win, int channel)
{
	// right = if (width%win =0 right =0; else, right = win- width%win, 
	// bottom = if (height%win =0 bottom =0; else, bottom = win - height%win
	int i, j;

	int widthPad = width + right;
	int heightPad = height + bottom;
	// copy original image 
	int startPos = 0;
	int startPosS = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst[startPos + i*width + j] = src[startPosS + i*widthPad + j];
		}
	}	
}

//get the average image 
void subtractImage(unsigned char* src, int* dst, int* dst2, int width, int height, int average)
{
	int i, j; 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst[i*width + j] = (int)src[i*width + j] - average;
			dst2[i*width + j] = dst[i*width + j] * dst[i*width + j]; 
		}
	}
}

// intergral image:
void integralImage(int* src, int* dst, int width, int height)
{
	int i, j;
	
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dst[i*width + j] = src[i*width + j];	
		}
	}

	for (i = 1; i < height; i++)
	{
		dst[i*width + 0] += dst[(i - 1)*width + 0];
	}
	for (j = 1; j < width; j++)
	{
		dst[0 + j] += dst[0 + j-1];
	}
	for (i = 1; i < height; i++)
	{
		for (j = 1; j < width; j++)
		{
			dst[i*width + j] += dst[(i-1)*width + j] + dst[i*width + j-1] - dst[(i - 1)*width + j -1];
		}
	}
}

// calculate local mean, local std
int  LocalMeanStd(int* padImgG, int* padImg2G, int* LocalMean, int* LocalStd, int width, int height, int win)
{
	int i, j;
	float tempVal, tempVal2; 
	int areaS = 2* win * 2* win; 
	int widthPad = width + 2 * win; 
	int maxRN = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			LocalMean[i*width + j] = (int)padImgG[i*widthPad + j]
				+ (int)padImgG[(i + 2 * win)*widthPad + j + 2 * win]
				- (int)padImgG[(i)*widthPad + j + 2 * win]
				- (int)padImgG[(i + 2 * win)*widthPad + j];
				tempVal = LocalMean[i*width + j] / (2*win); 
				tempVal2 = tempVal*tempVal;
			LocalStd[i*width + j] = (int)padImg2G[i*widthPad + j]
				+ (int)padImg2G[(i + 2 * win)*widthPad + j + 2 * win]
				- (int)padImg2G[(i)*widthPad + j + 2 * win]
				- (int)padImg2G[(i + 2 * win)*widthPad + j]
				- tempVal2;

			if (LocalStd[i*width + j] > maxRN)
				maxRN = LocalStd[i*width + j];
		}
	}
	return maxRN; 
}

void CalThreshold(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	)
{
	int i, j, index;
	float meanVal, stdVal, tempVal, meanRange, tempVal2; 
	float thresholdF1, thresholdF2; 
	//unsigned char* meanImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	//float* stdRImg = (float*)malloc(height*width* sizeof(float));
	unsigned char* tempImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* tempImg2 = (unsigned char*)malloc(height*width* sizeof(unsigned char));
	unsigned char* thrSau = (unsigned char*)malloc(height*width * sizeof(unsigned char));
	int stdAvg = 0, countStd =0 ;
	int stdAvgVal=0; 
	int tempStd, minStd = 255, maxStd=0; 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			tempStd= (int)sqrt(LocalStd[i*width + j])/(2*win);	
			//tempStd = (int)(LocalStd[i*width + j]);// / (2 * win);
			stdAvg += tempStd;
			if (minStd > tempStd)
				minStd = tempStd; 
			if (maxStd < tempStd);
				maxStd = tempStd;
		}
	}
	
	//stdAvgVal = sqrt(stdAvg) / (height*width *2*win); 
	stdAvgVal = stdAvg / (height*width);

	int nMinMax[2] = { 0,0 }; 
	int tempMaxRN = sqrt(maxRN / (2 * win * 2 * win)); // max std
	MinMaxImage(src, width, height, 4, nMinMax);
	if (tempMaxRN > (nMinMax[1] - nMinMax[0]) / 2)
		tempMaxRN = ( (nMinMax[1] - nMinMax[0]) / 2 + tempMaxRN)/2;

	
	tempMaxRN = 5 * stdAvgVal / 2; //51 //20* stdAvgVal/2; //51
	//tempMaxRN =  stdAvgVal / 2;

	//tempMaxRN = 3* stdAvgVal; //51
	//tempMaxRN = 7* stdAvgVal/2; //51
	//tempMaxRN = 14 * stdAvgVal / 2; //51
	float kValC = 5; 

	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			meanVal = average + (float)LocalMean[index] / (4 * win*win);
			meanImg[index] = (unsigned char) meanVal;
			//tempVal = kVal*(    sqrt((float)LocalStd[index] / maxRN) - 1);
			stdRImg[index] = 1- sqrt((float)LocalStd[index] / maxRN) ;  // positive value 
			//tempVal = kImg[index]*kVal*(1- sqrt((float)LocalStd[index] / maxRN)); // positive value 
			tempVal = kValC* kVal*(1 - sqrt((float)LocalStd[index] / maxRN)); // positive value 
			tempVal2 = (1 - sqrt((float)LocalStd[index] / maxRN));
			//meanRange = abs((float)meanImg[index] - 0) + 0.0f;  // consider that Min value =30; to avoid meanRange =0
			//meanRange = abs((float)meanImg[index] - (float)average) + 2.0f;  // consider that Min value =30; to avoid meanRange =0
			meanRange = tempMaxRN; // (nMinMax[1] - nMinMax[0]) / 2;// 64;// 184 / 3;// 46;// 184 / 3;// 46;
			//meanRange = 5*sqrt((float)LocalStd[index]) / (2 * win);
			thrSau[index] = meanImg[index] * (1 - tempVal);

			//thresholdF1 = (meanImg[index] * (1 + tempVal)); // greater than m 
			//thresholdF2 = (meanImg[index] * (1 - tempVal)); // smaller than m 
			//thresholdF1 = (meanImg[index]  + meanRange*tempVal); // greater than m 
			//thresholdF2 = (meanImg[index]  - meanRange*tempVal); // smaller than m 

			//thresholdF1 = meanImg[index] + meanRange*tempVal*1/2 + meanImg[index] * tempVal*1/2; // greater than m 
			//thresholdF2 = meanImg[index] - meanRange*tempVal*1/2 - meanImg[index] * tempVal * 1 / 2; // smaller than m 
			//
			//thresholdF1 = meanImg[index] + 5 * stdAvgVal / 2 *tempVal * 1 / 2 + meanImg[index] * tempVal * 1 / 2; // greater than m 
			//thresholdF2 = meanImg[index] - 5 * stdAvgVal / 2 *tempVal * 1 / 2 - meanImg[index] * tempVal * 1 / 2; // smaller than m 
			
			//thresholdF1 = meanImg[index] + 5 * stdAvgVal / 2 * tempVal ; // greater than m 
			//thresholdF2 = meanImg[index] - 5 * stdAvgVal / 2 * tempVal ; // smaller than m 

			//if (meanImg[index] <= 5 * stdAvgVal / 2)
			//{
			//	thresholdF1 = meanImg[index] +  meanImg[index]  * tempVal2; // greater than m 
			//	thresholdF2 = meanImg[index] -  meanImg[index]   * tempVal2; // smaller than m 

			//}
			//else if (255-meanImg[index] <= 5 * stdAvgVal / 2)
			//{
			//	thresholdF1 = meanImg[index] +  (255-meanImg[index])   * tempVal2; // greater than m 
			//	thresholdF2 = meanImg[index] -  (255 - meanImg[index])   * tempVal2; // smaller than m 

			//}
			//else 
			{


			thresholdF1 = meanImg[index] + 12 * stdAvgVal*kValC* kVal / 2 * tempVal2; // greater than m 
			thresholdF2 = meanImg[index] - 12 * stdAvgVal*kValC* kVal / 2 * tempVal2; // smaller than m 
			}

			if (thresholdF1 > 255)			
			{
				//printf("%f Over255 \n", thresholdF1);
				//printf("   %f  ,  %d   ", tempVal, meanImg[index]);
				thresholdF1 = 255; 
			}

			if (thresholdF2 < 0)
			{
				//printf("%f Less than 0 \n", thresholdF2);
				//printf("   %f  ,  %d   ", tempVal, meanImg[index]);
				thresholdF2 = 0;
			}
			thr1[index] = thresholdF1; // greater than m 
			thr2[index] = thresholdF2; // smaller than m 

			//thr1[index] = (unsigned char)(meanImg[index]   + 256*tempVal); // greater than m 
			//thr2[index] = (unsigned char)(meanImg[index]   - 256*tempVal); // smaller than m 

			if (src[index] > thr2[index])
				dst[index] = (unsigned char)255; 
			else
				dst[index] = (unsigned char)0;

			if (src[index] > thr1[index])
				tempImg[index] = (unsigned char)255;
			else
				tempImg[index] = (unsigned char)0;

			tempImg2[index] = src[index];			
			tempImg2[index] = (unsigned char)127;
			//if (LocalStd[index]*4 > maxRN )
			{
				
				if (src[index] > thresholdF1 && src[index] > thresholdF2) //thr1[index])
					tempImg2[index] = (unsigned char)255;
				if (src[index] < thresholdF2 && src[index] < thresholdF1)  //thr2[index])
					tempImg2[index] = (unsigned char)0;
				
			}

		}
	}
	// source image 
	WriteImage2Jpg(src, width, height, "./processDebug/0srcImg.png");
	// mean value 
	WriteImage2Jpg(meanImg, width, height, "./processDebug/0meanImg.png");
	// threshold low
	WriteImage2Jpg(dst, width, height, "./processDebug/0thresImg.png");
	// threshold high
	WriteImage2Jpg(tempImg, width, height, "./processDebug/0thresImgR.png");
	// threshold low
	WriteImage2Jpg(thr1, width, height, "./processDebug/0thres.png");
	// threshold high
	WriteImage2Jpg(thr2, width, height, "./processDebug/0thresR.png");

	WriteImage2Jpg(thrSau, width, height, "./processDebug/0thresSau.png");

	

	// only extreme area is black and white, middle is gray 127
	WriteImage2Jpg(tempImg2, width, height, "./processDebug/0thresImgPN.png");
	//free(meanImg);
	//free(stdRImg);
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			if (src[index] > meanImg[index])
				tempImg[index] = (unsigned char)255;
			else
				tempImg[index] = (unsigned char)0;
		}
	}
	// use mean filter to fitler out the foreground and background 
	// --> mean value is good in extreme area 
	WriteImage2Jpg(tempImg, width, height, "./processDebug/0thresImgMeanR.png");
	free(tempImg);	
	free(tempImg2);
}

int range(int x, int y, int width, int height)
{
	if (x > width || x < 0)
		return 0; 
	if (y > height || y < 0)
		return 0;
	return 1; 
}


void FilterForeGround(unsigned char* srcdst, unsigned char* ref,
	int width, int height,
	int win)
{
	//scanning to remove the noise :
	//--> scan top - bottom, left - right.
	//	if found a component in a line
	//		--> make a threshold by left and right pixels
	//		--> if inside that line, there is pixels brighter than
	//		threshold--> set it as not the correct one.
	//		--> remove the wrong one.
	int i, j, i1, j1, index, index0, pos;
	int isStart, isStop, iCount, pStart, pStop; 
	int breakloop = 0; 
	index = 0; 
	unsigned char* dstProcess = (unsigned char*)malloc(height*width * sizeof(unsigned char));  // text threshold 
	unsigned char* temp = (unsigned char*)malloc(win/2*win/2 * sizeof(unsigned char));  // text threshold 
	// copy back the current dst; 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			dstProcess[index] = srcdst[index];
			index++;
		}
	}

	for (index0 =0; index0<height*width; index0++)
	{
		iCount = 0; 
		index = 0; 
		
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{				
				// check every pixels
				if (dstProcess[index] == 0)
				{
					breakloop = 0;
					for (i1 = i - win; i1 < i + win; i1++)
					{
						if (breakloop == 1)
							break; 
						for (j1 = j - win; j1 < j + win; j1++)
						{
							if (breakloop == 1)
								break;
							if (range(j1, i1, width, height))// in range 
							{
								pos = i1*width + j1; 
								if (dstProcess[pos]>0) // not the black one 
								//if (srcdst[pos]>0) // not the black one 
									if (ref[pos] < ref[index]-1)  // if found smaller, then current point is not correct
									//if (ref[pos] < ref[index])  // if found smaller, then current point is not correct

									{
										dstProcess[index] = 255; 
										breakloop = 1;
										iCount ++;
										break; 
									}
							}
						}
					}
				}
				index++;
				
			}
		}
		//iCount = 0;

		/*index = 0;
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				srcdst[index] = dstProcess[index];
				index++;
			}
		}*/

		if (iCount == 0)
			break;
	}

	// copy back to the source 
	index = 0;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			srcdst[index] =  dstProcess[index];
			index++;
		}
	}
	
	free(dstProcess);
	free(temp);
	
}

void CalThresholdBGFG2(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	)
{
	int i, j, i1, j1, index, index0;
	float meanVal, stdVal, tempVal;
	//unsigned char* meanImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));

	int* signVal = (int*)malloc(height*width* sizeof(int));
	int* upT = (int*)malloc(height*width* sizeof(int));


	int heightPad = height + 2 * win;
	int widthPad = width + 2 * win;

	unsigned char* lowImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* FG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* BG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* FGBG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* lowImgP = (unsigned char*)malloc(heightPad*widthPad* sizeof(unsigned char));  // text threshold 
	unsigned char* temp = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* temp2 = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 

	int* loT = (int*)malloc(heightPad*widthPad* sizeof(int));
	int* loTG = (int*)malloc(heightPad*widthPad* sizeof(int));
	int hist[3];
	int cFG, cBG, countFG, countBG;


	//float* stdRImg = (float*)malloc(height*width* sizeof(float));
	// thr1 > thr2
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			dst[index] = (unsigned char)127;
			signVal[index] = 0;
			FG[index] = 0;
			BG[index] = 255;
			FGBG[index] = 0;
			if (src[index] > thr2[index] && src[index] > thr1[index])
			{
				dst[index] = (unsigned char)255;
				signVal[index] = 2;   // use the smaller threshold BG
				lowImg[index] = 1;
				BG[index] = src[index];
				FGBG[index] += 2;
			}
			else if (src[index] <= thr2[index] && src[index] <= thr1[index]) // smaller than both threshold // that is the text 
			{
				dst[index] = (unsigned char)0;
				signVal[index] = 1;// use higher threshold  FG
				lowImg[index] = 0;
				FG[index] = src[index];
				FGBG[index] += 1;

			}
		}
	}
	//calculate the result image: which area: larger and smaller than both 2 threshold. 

	WriteImage2Jpg(dst, width, height, "./processDebug/1thresImgNew.png");
	//free(meanImg);
	//free(stdRImg);

	int* bFG = (int*)malloc(height*width* sizeof(int));
	int* bBG = (int*)malloc(height*width* sizeof(int));
	for (i = 0; i < height*width; i++)
	{
		bFG[i] = -1;
		bBG[i] = -1;
	}

	// scan loop 1: all point have the connection with before point: 
	int step = 8;// 8;// 128;// 32; (step >=2)
	int thresPoint =  step*step / (4 * 4);  //step*step /10 ;// step*step / (4 * 4);
	// example: 8x8 / 4*4 = 64/1 = 4 times
	// block 8x8 to check the threshold. non overlap block. 
	// if block size = 1 

	// check block by block
	int minthres = 1; 
	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			index = i*width + j;
			countFG = 0;
			countBG = 0;
			cFG = 0;
			cBG = 0;
			hist[0] = hist[1] = hist[2] = 0;
			// at block i,j: 
			for (i1 = 0; i1 < step; i1++) //height
			{
				for (j1 = 0; j1 < step; j1++)
				{
					//if (i*step + i1 >= height || j*step + j1 >= width)
					//	continue; 
					index0 = index + i1* width + j1; // current position 
													 // if this pixel is background or foreground --> accumulate it then calculate average of FG and BG. 
													 // then this region using the upper threshold or lower threshold. 
													 // if mean of this region is closer to BG or FG --> define the threshold for region. 
													 // if the region has no information: no BG or FG: --> find: until reach the BG and foreground: 
					if (signVal[index0] > 0)	 // if this pixel is background or foreground --> accumulate it then calculate average of FG and BG. 
					{
						if (signVal[index0] == 1)
						{
							cFG += src[index0];      // found FG and accumulate 
							countFG++;
						}
						if (signVal[index0] == 2)
						{
							cBG += src[index0];      // found BG and accumulate
							countBG++;
						}
					}
				}
			}   // complete counting in this block: 8x8 = 64 pixels
			//if (countFG > thresPoint && countBG> minthres) // if this block has countFG >4 and BG> 1 
			if (countFG >= minthres) // if this block has countFG >4 and BG> 1 
			{
				cFG /= countFG;
				bFG[i / step* (width / step) + j / step] = cFG;  // use this block as FG. 
			}
			//if (countBG > thresPoint && countFG> minthres)
			if (countBG >= minthres)
			{
				cBG /= countBG;
				bBG[i / step* (width / step) + j / step] = cBG;  // use this average value as BG for this block 
			}

			//if (countBG > thresPoint && countFG > thresPoint)  // if both has FG and BG: > 4
			if (countBG >= minthres && countFG >= minthres)  // if both has FG and BG: >=1  
			{
				//cFG /= countFG;
				//cBG /= countBG;
				//if (countBG + countFG < step*step)  // if the block has no pixel to consider. 
				{
					for (i1 = 0; i1 < step; i1++) //height
					{
						for (j1 = 0; j1 < step; j1++)
						{
					//		if (i*step + i1 >= height || j*step + j1 >= width)
					//			continue;
							index0 = index + i1* width + j1; // current position . choose closer value: FG or BG for the pixel
							if (dst[index0] == 127)
							{
								//if ((int)src[index0] - cFG < (int)src[index0] - cBG)
								//	dst[index0] = 0;
								//else
								//	dst[index0] = 255;

								if (abs((int)src[index0] - cFG) < abs((int)src[index0] - cBG))
									dst[index0] = 0;
								else
									dst[index0] = 255;
							}
						}
					}
				}
			}
			// how about the case: only BG and only FG in a block. 
		}
	}
	// now consider the non extreme pixels or block: 
	WriteImage2Jpg(dst, width, height, "./processDebug/1thresImgNewL1.png");

	for (i = 0; i < height / step; i++)
	{
		for (j = 0; j < width / step; j++)
		{
			// for each block: 
			index = i*(width / step) + j;
			if (bBG[index] == -1) // not yet define: the middle area, no foreground and background. 
				temp[index] = 127; // set as middle to wait 
			else
			{
				
				// normalize the temp value = bBG // trim the wrong value out side range 0 --> 255. (maybe redundant)
				if (bBG[index] >= 255)  // which is clearly FG // all block is BG
					temp[index] = 255; // has defined 
				else if (bBG[index] <= 0) // which is not clear --> redundant this one. (==-1)
					temp[index] = 0;
				else 
					temp[index] = (unsigned char)bBG[index]; // middle value BG. 
			}

			// consider the FG. 
			if (bFG[index] == -1)
			{
				temp2[index] = 127;
			}
			else
			{
				if (bFG[index] >= 255)
					temp2[index] = 255;
				else if (bFG[index] <= 0)
					temp2[index] = 0;
				else
					temp2[index] = (unsigned char)bFG[index];
			}

		}
	}
	WriteImage2Jpg(temp, width / step, height / step, "./processDebug/1SmallImgBG.png");
	WriteImage2Jpg(temp2, width / step, height / step, "./processDebug/1SmallImgFG.png");

	// Find around the A and B 
	// if found A or B: check the block in neighbor
	int indexJ, indexI; 
	int* checkX = (int*)malloc(height*width* sizeof(int));
	int* checkY = (int*)malloc(height*width* sizeof(int));
	int numCheck = -1; 
	int indexIR, indexJR;
	int tempBG = 0, tempCount = 0;
	int tempFG = 0;
	
	while (1) // check the block stepxstep whole the image 
	{
		numCheck = -1;  // check each block. 
		for (i = 0; i < height / step; i++)
		{
			for (j = 0; j < width / step; j++)
			{
				index = i*(width / step) + j;
				// check all block 
				if (bBG[index] != -1) // if a block is found, find around it and mark it. 
				{
					indexJ = j - 1;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step)) // maybe block effect here 
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
							// record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;				
						}
					}
					indexJ = j + 1;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j - 1;
					indexI = i;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j + 1;
					indexI = i;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j - 1;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j + 1;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bBG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;
						}
					}
				}

			}
		}
		if (numCheck == -1) // if there is no unchecked blocks --> finish 
			break; 
		// with the check list, check each point, update // numCheck = 0 --> 1 check
	
		tempBG = 0; tempCount = 0;
		for (index = 0; index <= numCheck; index++) // each block: check the around blocks
		{
			indexJ = checkX[index];
			indexI = checkY[index];
			tempBG = 0; tempCount = 0;
			if (bBG[indexI*(width / step) + indexJ] == -1)
			{
				indexJR = indexJ - 1;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ - 1;
				indexIR = indexI;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ - 1;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bBG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bBG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}

				// calculate the average BG: 
				if (tempCount >= 1)
				{
					bBG[indexI*(width / step) + indexJ] = tempBG / tempCount;
				}
			}
		}
	}
	//
	// loop for foreground FG
	while (1)
	{
		numCheck = -1;
		for (i = 0; i < height / step; i++)
		{
			for (j = 0; j < width / step; j++)
			{
				index = i*(width / step) + j;
				if (bFG[index] != -1)
				{
					indexJ = j - 1;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
							// record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;				
						}
					}
					indexJ = j + 1;
					indexI = i - 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j - 1;
					indexI = i;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j + 1;
					indexI = i;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j - 1;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later.
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;							
						}
					}
					indexJ = j + 1;
					indexI = i + 1;
					if (range(indexJ, indexI, width / step, height / step))
					{
						if (bFG[indexI*(width / step) + indexJ] == -1)
						{// update this point. 
						 // record and check later. 						
							numCheck++;
							checkX[numCheck] = indexJ;
							checkY[numCheck] = indexI;
						}
					}
				}
			}
		}
		if (numCheck == -1)
			break; 
		// with the check list, check each point, update // numCheck = 0 --> 1 check
		
		tempBG = 0; tempCount = 0;
		for (index = 0; index <= numCheck; index++)
		{
			indexJ = checkX[index];
			indexI = checkY[index];
			tempBG = 0; tempCount = 0;
			if (bFG[indexI*(width / step) + indexJ] == -1)
			{
				indexJR = indexJ - 1;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI - 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ - 1;
				indexIR = indexI;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ - 1;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}
				indexJR = indexJ + 1;
				indexIR = indexI + 1;
				if (range(indexJR, indexIR, width / step, height / step))
				{
					if (bFG[indexIR*(width / step) + indexJR] != -1)
					{// use this one 
						tempBG += bFG[indexIR*(width / step) + indexJR];
						tempCount++;
					}
				}

				// calculate the average BG: 
				if (tempCount >= 1)
				{
					bFG[indexI*(width / step) + indexJ] = tempBG / tempCount;
				}
			}
		}
	}
	

	for (i = 0; i < height / step; i++)
	{
		for (j = 0; j < width / step; j++)
		{
			index = i*(width / step) + j;
			if (bBG[index] == -1)
				temp[index] = 127;
			else
				temp[index] = (unsigned char)bBG[index];

			if (bFG[index] == -1)
				temp2[index] = 127;
			else
				temp2[index] = (unsigned char)bFG[index];

		}
	}

	WriteImage2Jpg(temp, width / step, height / step, "./processDebug/1SmallImgBG2_.png");
	WriteImage2Jpg(temp2, width / step, height / step, "./processDebug/1SmallImgFG2_.png");

	//
	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			index = i*width + j;

			cBG = bBG[i / step* (width / step) + j / step];
			cFG = bFG[i / step* (width / step) + j / step];
			// at block i,j: 
			for (i1 = 0; i1 < step; i1++) //height
			{
				for (j1 = 0; j1 < step; j1++)
				{
				//	if (i*step + i1 >= height || j*step + j1 >= width)
				//		continue;
					index0 = index + i1* width + j1; // current position 
					if (dst[index0] == 127)
					{
						if (abs((int)src[index0] - cFG) < abs((int)src[index0] - cBG))
							dst[index0] = 0;
						else
							dst[index0] = 255;
					}
				}
			}
		}
	}

	//WriteImage2Jpg(dst, width, height, "./input/thresImgNew2.png");
	
	WriteImage2Jpg(dst, width, height, "./processDebug/1thresImgNew2.png");
	//free(meanImg);
	//free(stdRImg);

	// if all black --> threshold black, // if all white --> threshold white
	// if around is black --> around using threshold: 


	MakepadImage(lowImg, lowImgP, width, height, win);

	for (i = 0; i < heightPad; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			index = i*widthPad + j;
			loT[index] = lowImgP[index];
		}
	}
	integralImage(loT, loTG, widthPad, heightPad);

	// count number of black/white 

	free(signVal);
	free(upT);
	free(lowImg);
	free(FG);
	free(BG);
	free(FGBG);
	free(lowImgP);
	free(temp);
	free(temp2);
	free(loT);
	free(loTG);
	free(bFG);
	free(bBG);

	free(checkX);
	free(checkY);
	//free(updateVal);
	//free(visited);
}


void CalThresholdBGFG(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	)
{
	int i, j, i1, j1, index, index0;
	float meanVal, stdVal, tempVal;
	//unsigned char* meanImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));

	int* signVal = (int*)malloc(height*width* sizeof(int));
	int* upT = (int*)malloc(height*width* sizeof(int));


	int heightPad = height + 2 * win;
	int widthPad = width + 2 * win;

	unsigned char* lowImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* FG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* BG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* FGBG = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* lowImgP = (unsigned char*)malloc(heightPad*widthPad* sizeof(unsigned char));  // text threshold 
	unsigned char* temp = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* temp2 = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 

	int* loT = (int*)malloc(heightPad*widthPad* sizeof(int));
	int* loTG = (int*)malloc(heightPad*widthPad* sizeof(int));
	int hist[3];
	int cFG, cBG, countFG, countBG; 


	//float* stdRImg = (float*)malloc(height*width* sizeof(float));
	// thr1 > thr2
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			dst[index] = (unsigned char)127;
			signVal[index] = 0;
			FG[index] = 0; 
			BG[index] = 255;
			FGBG[index] = 0;
			if (src[index] > thr2[index] && src[index] > thr1[index])
			{
				dst[index] = (unsigned char)255;
				signVal[index] = 2;   // use the smaller threshold BG
				lowImg[index] = 1;
				BG[index] = src[index];
				FGBG[index] += 2;
			}
			else if (src[index] <= thr2[index] && src[index] <= thr1[index])
			{
				dst[index] = (unsigned char)0;
				signVal[index] = 1;// use higher threshold  FG
				lowImg[index] = 0;
				FG[index] = src[index];
				FGBG[index] += 1;

			}
		}
	}
	WriteImage2Jpg(dst, width, height, "./processDebug/thresImgNew.png");
	//free(meanImg);
	//free(stdRImg);

	int* bFG = (int*)malloc(height*width* sizeof(int));
	int* bBG = (int*)malloc(height*width* sizeof(int));
	for (i = 0; i < height*width; i++)
	{
		bFG[i] = -1; 
		bBG[i] = -1;
	}

	// scan loop 1: all point have the connection with before point: 
	int step = 16; 
	int thresPoint = (step*step) / (4 * 4);// 4;
	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			index = i*width + j;
			countFG = 0;
			countBG = 0;
			cFG = 0;
			cBG = 0;
			hist[0] = hist[1] = hist[2] = 0;
			// at block i,j: 
			for (i1 = 0; i1 < step; i1++) //height
			{
				for (j1 = 0; j1 < step; j1++)
				{
					index0 = index + i1* width + j1; // current position 
					// if this pixel is background or foreground --> accumulate it then calculate average of FG and BG. 
					// then this region using the upper threshold or lower threshold. 
					// if mean of this region is closer to BG or FG --> define the threshold for region. 
					// if the region has no information: no BG or FG: --> find: until reach the BG and foreground: 
					if (signVal[index0] > 0)
					{
						if (signVal[index0] == 1)
						{
							cFG += src[index0];
							countFG++;
						}
						if (signVal[index0] == 2)
						{
							cBG += src[index0];
							countBG++;
						}
					}
				}
			}
			if (countFG > thresPoint)
			{
				cFG /= countFG;
				bFG[i / step* (width / step) + j/step] = cFG; 
			}
			if (countBG > thresPoint)
			{
				cBG /= countBG;
				bBG[i / step* (width / step) + j/step] = cBG;
			}

			if (countBG > thresPoint && countFG > thresPoint)
			{
				//cFG /= countFG;
				//cBG /= countBG;
				if (countBG + countFG < step*step)  // if the block has no pixel to consider. 
				{
					for (i1 = 0; i1 < step; i1++) //height
					{
						for (j1 = 0; j1 < step; j1++)
						{
							index0 = index + i1* width + j1; // current position 
							if (dst[index0] ==127)
							{
								if (abs((int)src[index0] - cFG) < abs((int)src[index0] - cBG))
									dst[index0] = 0;
								else
									dst[index0] = 255;
							}
						}
					}
				}
			}
		}
	}

	for (i = 0; i < height / step; i++)
	{
		for (j = 0; j < width / step; j++)
		{
			index = i*(width / step) + j; 
			if (bBG[index] == -1)
				temp[index] = 127;
			else 
				temp[index] = (unsigned char)bBG[index];

			if (bFG[index] == -1)
				temp2[index] = 127;
			else
				temp2[index] = (unsigned char)bFG[index];

		}
	}
	WriteImage2Jpg(temp, width/step, height/step, "./processDebug/SmallImgBG.png");
	WriteImage2Jpg(temp2, width / step, height / step, "./processDebug/SmallImgFG.png");



	int i_1, j_1, i_2, j_2, d, isFound;
	
	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			index = i*width + j;
			
			if (bFG[i / step* (width / step) + j/step] == -1)
			{
				isFound = 0;
					for (d = 0; d < 100; d++)
					{
						if (i / step* (width / step) + j/step + d*(width / step) + d >(width/step)*(height/step))
							break; 
						else 
							if (bFG[i / step* (width / step) + j/step + d*(width/step)+ d] != - 1)
							{
								bFG[i / step* (width / step) + j/step] = bFG[i / step* (width / step) + j/step + d*(width / step) + d];
								isFound = 1; 
								break; 
							}
						
					}
				if (isFound ==0)
					for (d = 0; d > -100; d--)
					{
						if (i / step* (width / step) + j/step + d*(width / step) + d < 0)
							break; 
						else 
							if (bFG[i / step* (width / step) + j/step + d*(width / step) + d] != - 1)
							{
								bFG[i / step* (width / step) + j/step] = bFG[i / step* (width / step) + j/step + d*(width / step) + d];
								isFound = 1; 
								break;
							}
					}
				if (isFound == 0)
					for (d = 0; d > -100; d--)
					{
						if (i / step* (width / step) + j/step - d*(width / step) + d < 0)
							break;
						else
							if (bFG[i / step* (width / step) + j/step - d*(width / step) + d] != - 1)
							{
								bFG[i / step* (width / step) + j/step] = bFG[i / step* (width / step) + j/step - d*(width / step) + d];
								isFound = 1; 
								break;
							}
					}

			}

			if (bBG[i / step* (width / step) + j/step] == -1)
			{
				isFound = 0;
				for (d = 0; d < 100; d++)
				{
					if (i / step* (width / step) + j/step + d*(width / step) + d >(width / step)*(height / step))
						break;
					else
						if (bBG[i / step* (width / step) + j/step + d*(width / step) + d] != - 1)
						{
							bBG[i / step* (width / step) + j/step] = bBG[i / step* (width / step) + j/step + d*(width / step) + d];
							isFound = 1;
							break;
						}

				}
				if (isFound ==0)
				for (d = 0; d > -100; d--)
				{
					if (i / step* (width / step) + j/step + d*(width / step) + d < 0)
						break;
					else
						if (bBG[i / step* (width / step) + j/step + d*(width / step) + d] != - 1)
						{
							bBG[i / step* (width / step) + j/step] = bBG[i / step* (width / step) + j/step + d*(width / step) + d];
							isFound = 1;
							break;
						}
				}
				if (isFound == 0)
					for (d = 0; d > -100; d--)
					{
						if (i / step* (width / step) + j/step - d*(width / step) + d < 0)
							break;
						else
							if (bBG[i / step* (width / step) + j/step - d*(width / step) + d] != - 1)
							{
								bBG[i / step* (width / step) + j/step] = bBG[i / step* (width / step) + j/step - d*(width / step) + d];
								isFound = 1;
								break;
							}
					}
			}

		}
	}
	
	/*
	for (i = 0; i < height / step; i++)
	{
		for (j = 0; j < width / step; j++)
		{
			index = i*(width / step) + j;
			if (bBG[index] == -1)
			{
				isFound = 0;
				for (d = 0; d < 100; d++)
				{
					if (index + d*(width / step) + d >(width / step)*(height / step))
						break;
					else
						if (bBG[index + d*(width / step) + d] != - 1)
						{
							bBG[index] = 255;// bBG[index + d*(width / step) + d];
							isFound = 1;
							break;
						}
				}
				
				if (isFound == 0)
					for (d = 0; d > -100; d--)
					{
						if (index + d*(width / step) + d<0 )
							break;
						else
							if (bBG[index + d*(width / step) + d] = !- 1)
							{
								bBG[index] = bBG[index + d*(width / step) + d];
								isFound = 1;
								break;
							}
					}
				if (isFound == 0)
					for (d = 0; d > -100; d--)
					{
						if (index + d*(width / step) + d <0)
							break;
						else
							if (bBG[index + d*(width / step) - d] = !- 1)
							{
								bBG[index] = bBG[index + d*(width / step) - d];
								isFound = 1;
								break;
							}
					}
					
			}
		}
	}
	*/

			


	for (i = 0; i < height / step; i++)
	{
		for (j = 0; j < width / step; j++)
		{
			index = i*(width / step) + j;
			if (bBG[index] == -1)
				temp[index] = 127;
			else
				temp[index] = (unsigned char)bBG[index];

			if (bFG[index] == -1)
				temp2[index] = 127;
			else
				temp2[index] = (unsigned char)bFG[index];

		}
	}

	WriteImage2Jpg(temp, width / step, height / step, "./processDebug/SmallImgBG2.png");
	WriteImage2Jpg(temp2, width / step, height / step, "./processDebug/SmallImgFG2.png");

	for (i = 0; i < height; i += step)
	{
		for (j = 0; j < width; j += step)
		{
			index = i*width + j;
			
			cBG = bBG[i / step* (width / step) + j/step] ;
			cFG = bFG[i / step* (width / step) + j/step] ;
			// at block i,j: 
			for (i1 = 0; i1 < step; i1++) //height
			{
				for (j1 = 0; j1 < step; j1++)
				{
					index0 = index + i1* width + j1; // current position 
					if (dst[index0] == 127)
					{
						if (abs((int)src[index0] - cFG) < abs((int)src[index0] - cBG))
							dst[index0] = 0;
						else
							dst[index0] = 255;
					}
				}
			}
		}
	}


	
	WriteImage2Jpg(dst, width, height, "./processDebug/thresImgNew2.png");
	//free(meanImg);
	//free(stdRImg);

	// if all black --> threshold black, // if all white --> threshold white
	// if around is black --> around using threshold: 


	MakepadImage(lowImg, lowImgP, width, height, win);

	for (i = 0; i < heightPad; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			index = i*widthPad + j;
			loT[index] = lowImgP[index];
		}
	}
	integralImage(loT, loTG, widthPad, heightPad);

	// count number of black/white 


}


void CalThresholdNew(unsigned char* src, unsigned char* dst,
	unsigned char* kImg, unsigned char* thr1, unsigned char* thr2,
	int* LocalMean, int* LocalStd, int width, int height, int maxRN,
	int win, float kVal, int average,
	unsigned char* meanImg, float* stdRImg
	)
{
	int i, j, index;
	float meanVal, stdVal, tempVal;
	//unsigned char* meanImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));

	int* signVal = (int*)malloc(height*width* sizeof(int));
	int* upT = (int*)malloc(height*width* sizeof(int));
	
	
	int heightPad = height + 2 * win;
	int widthPad = width + 2 * win;

	unsigned char* lowImg = (unsigned char*)malloc(height*width* sizeof(unsigned char));  // text threshold 
	unsigned char* lowImgP = (unsigned char*)malloc(heightPad*widthPad* sizeof(unsigned char));  // text threshold 

	int* loT = (int*)malloc(heightPad*widthPad* sizeof(int));
	int* loTG = (int*)malloc(heightPad*widthPad* sizeof(int));
	int hist[3];


	//float* stdRImg = (float*)malloc(height*width* sizeof(float));
	// thr1 > thr2
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			dst[index] = (unsigned char)127;
			signVal[index] = 0; 

			if (src[index] > thr2[index]&& src[index] > thr1[index])
			{
				dst[index] = (unsigned char)255;
				signVal[index] = 2;   // use the smaller threshold 
				lowImg[index] = 1;
			}
			else if (src[index] <= thr2[index] && src[index] <= thr1[index])
			{
				dst[index] = (unsigned char)0;
				signVal[index] = 1;// use higher threshold 
				lowImg[index] = 0;
			}
		}
	}
	WriteImage2Jpg(dst, width, height, "./processDebug/thresImgNew.png");
	//free(meanImg);
	//free(stdRImg);

	


	// scan loop 1: all point have the connection with before point: 
	hist[0] = hist[1] = hist[2] = 0; 
	for (i = 0+1; i < height-1; i++)
	{
		for (j = 0+1; j < width-1; j++)
		{
			index = i*width + j;
			hist[0] = hist[1] = hist[2] = 0;
			if (dst[index] == 127)
			{				
				hist[signVal[index - width - 1]]++;
				hist[signVal[index - width ]]++;
				hist[signVal[index - width + 1]]++;

				hist[signVal[index - 1]]++;
				hist[signVal[index ]]++;
				hist[signVal[index + 1]]++;

				hist[signVal[index + width - 1]]++;
				hist[signVal[index + width]]++;
				hist[signVal[index + width + 1]]++;
			}
			if (hist[2] != 0 || hist[1] != 0)
			{
				if (src[index] >= (hist[1] * (int)thr1[index] + hist[2] * (int)thr2[index]) / ((int)hist[1] + (int)hist[2]))
					dst[index] = 235;
				else
					dst[index] = 20; 
			}
			else  // both hist 1, 2 = 0; 
			{

			}
		}
	}
	WriteImage2Jpg(dst, width, height, "./processDebug/thresImgNew2.png");
	//free(meanImg);
	//free(stdRImg);

	// if all black --> threshold black, // if all white --> threshold white
	// if around is black --> around using threshold: 
	

	MakepadImage(lowImg, lowImgP, width, height, win);

	for (i = 0; i < heightPad; i++)
	{
		for (j = 0; j < widthPad; j++)
		{
			index = i*widthPad + j;
			loT[index] = lowImgP[index];
		}
	}
	integralImage(loT, loTG, widthPad, heightPad);

	// count number of black/white 
	
	
}


int xGradient(unsigned char* src, int width, int height, int x, int y)
{
	int i, j, index; 
	int gX;
	index = y*width + x; 
	gX = src[(y - 1)*width + x - 1] - src[(y - 1)*width + x + 1]
		+ 2 * src[(y)*width + x - 1] - 2 * src[(y)*width + x + 1]
		+ src[(y + 1)*width + x - 1] - src[(y + 1)*width + x + 1];
	return gX; 
}

int yGradient(unsigned char* src, int width, int height, int x, int y)
{
	int i, j, index;
	int gY;
	index = y*width + x;
	gY =      src[(y - 1)*width + x - 1]  + 2 * src[(y - 1)*width + x  ] + src[(y - 1)*width + x + 1]
		-     src[(y + 1)*width + x - 1]  - 2 * src[(y + 1)*width + x  ] - src[(y + 1)*width + x + 1];
	return gY;
}

int sobelFilter(unsigned char* src, unsigned char* dst, int width, int height)
{
	int i, j, index; 
	int maxSobel =0; 
	int gX, gY, gS; 
	i = 0; j = 0;  dst[i*width + j] = 0;
	i = 0; j = width-1;  dst[i*width + j] = 0;
	i = height-1; j = 0;  dst[i*width + j] = 0;
	i = height - 1; j = width - 1;  dst[i*width + j] = 0;

	for (i = 0+1; i < height-1; i++)
	{
		for (j = 0+1; j < width-1; j++)
		{
			
			gX = xGradient(src, width, height, j, i);
			gY = xGradient(src, width, height, j, i);
			//gS = sqrt(gX*gX + gY*gY);
			gS = abs(gX) + abs(gY);
			if (maxSobel < gS)
				maxSobel = gS;
			if (gS > 255)
				gS = 255; 
			dst[i*width + j] = gS; 
		}
	}
	return maxSobel; 
}

int sobelImage(unsigned char* src, unsigned char* dst, unsigned char* kImg, int width, int height, int th1, int th2)
{
	int i, j, index; 
	//int k0 = 1, k1 = 2, k2 = 3; 
	//int k0 = 1, k1 = 2, k2 = 4; // seem to be adaptive but sometime too thick 
	//int k0 = 2, k1 = 2, k2 = 2; 
	//int k0 = 4, k1 = 4, k2 = 4; // higher value --> less noise
	int k0 = 5, k1 = 5, k2 = 5; // higher value --> less noise
	//int k0 = 1, k1 = 3, k2 = 5; // higher value --> less noise
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j; 
			if (src[index] > th1)
			{
				dst[index] = 255;
				kImg[index] = k0;
			}
			else if (src[index] > th2)
			{
				dst[index] = 1;
				kImg[index] = k1;
			}
			else
			{
				dst[index] = 0; 
				kImg[index] = k2;
			}
		}
	}
	return 0; 
}


//int minVal(int x1, int x2, int x3, int x4)
//{
//	int minValue =x1; 
//	if (minValue > x2)
//		minValue = x2; 
//	if (minValue > x3)
//		minValue = x3;
//	if (minValue > x4)
//		minValue = x4;
//	return minValue; 
//}

int minVal(int x1, int x2)
{
	int minValue = x1;
	if (minValue > x2)
		minValue = x2;
	return minValue;
}

int signImage(unsigned char* src, unsigned char* meanImg, float* stdRImg, 
	unsigned char* kImg, unsigned char* thr, int width, int height, unsigned char* thr1, unsigned char* thr2, float kVal )
{
	int i, j, index;
	int k0 = 1, k1 = 2, k2 = 3;
	float tempVal;
	
	int* signPos = (int*)malloc(height*width* sizeof(int));
	int* signNev = (int*)malloc(height*width* sizeof(int));
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			signNev[index] = MAX_INT;
			signPos[index] = MAX_INT;
		}
	}

	// check sign: k0 --> consider later, k1 --> close to mean --> 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			if (kImg[index] == k0)
			{
				tempVal = kImg[index] * kVal*(stdRImg[index]);
				thr1[index] = (unsigned char)(meanImg[index] * (1 - tempVal)); 
				thr2[index] = (unsigned char)(meanImg[index] * (1 + tempVal));
				if (src[index] < meanImg[index])
				{					
					//stdRImg[index] = sqrt((float)LocalStd[index] / maxRN) - 1;					
					thr[index] = thr2[index];
					signPos[index] = 1; 
				}
				else if (src[index] >= meanImg[index])
				{
					//stdRImg[index] = sqrt((float)LocalStd[index] / maxRN) - 1;					
					thr[index] = thr1[index];
					signNev[index] = 1;
				}
			}
		}
	}
	// loop 2: around pixel scan top left to bottom down
	int minKRoundP =MAX_INT, minKRoundN = MAX_INT; 
	for (i = 0+1; i < height-1; i++)
	{
		for (j = 0+1; j < width-1; j++)
		{
			index = i*width + j;
			if (kImg[index] != k0)
			{
				minKRoundP = minVal(signPos[index - 1], signPos[index - width]);
				if (minKRoundP < signPos[index])
					signPos[index] = minKRoundP + 1;
				minKRoundN = minVal(signNev[index - 1], signNev[index - width]);
				if (minKRoundN < signNev[index])
					signNev[index] = minKRoundN + 1;
			}
		}
		for (j = width - 1 - 1; j>= 0 + 1; j--)
		{
			index = i*width + j;
			if (kImg[index] != k0)
			{
				minKRoundP = minVal(signPos[index + 1], signPos[index - width]);
				if (minKRoundP < signPos[index])
					signPos[index] = minKRoundP + 1;
				minKRoundN = minVal(signNev[index + 1], signNev[index - width]);
				if (minKRoundN < signNev[index])
					signNev[index] = minKRoundN + 1;
			}
		}
	}
	// loop 3: scan bottom right to top left: 
	for (i = height - 1 - 1;  i >= 0 + 1;  i--)
	{
		for (j = width - 1 - 1;  j >= 0 + 1; j--)
		{
			index = i*width + j;
			if (kImg[index] != k0)
			{
				minKRoundP = minVal(signPos[index + 1], signPos[index + width]);
				if (minKRoundP < signPos[index])					
					signPos[index] = minKRoundP + 1;
				minKRoundN = minVal(signNev[index + 1], signNev[index + width]);
				if (minKRoundN < signNev[index])					
					signNev[index] = minKRoundN + 1;
			}
		}
		for (j = 0 + 1; j < width - 1 ;  j++)
		{
			index = i*width + j;
			if (kImg[index] != k0)
			{
				minKRoundP = minVal(signPos[index - 1], signPos[index + width]);
				if (minKRoundP < signPos[index])
					signPos[index] = minKRoundP + 1;
				minKRoundN = minVal(signNev[index - 1], signNev[index + width]);
				if (minKRoundN < signNev[index])
					signNev[index] = minKRoundN + 1;
			}
		}
	}
	// last loop for make sign for the remain pixel 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			if (kImg[index] != k0)
			{
				if (signNev[index] < signPos[index])
					thr[index] = thr1[index];
				else
					thr[index] = thr2[index];
			}
		}
	}

	free(signPos);
	free(signNev);
	
	return 0; 

}

void threshImg(unsigned char* src, unsigned char* thr, unsigned char* dst, int width, int height)
{
	int i, j, index; 
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			index = i*width + j;
			if (src[index] < thr[index])
				dst[index] = 0;
			else
				dst[index] = 255;
		}
	}
}


//void WriteImage2Jpg(unsigned char* src, int width, int height, char* filename )
//{
//	int y; 
//	
//	IplImage* src_gray = cvCreateImage(CvSize(width, height), IPL_DEPTH_8U, 1);
//	
//	for (y = 0; y < src_gray->height; y++)
//		memcpy(src_gray->imageData + y*src_gray->widthStep, src + y*src_gray->width, src_gray->width);
//
//	cvSaveImage(filename, src_gray);
//
//	cvReleaseImage(&src_gray);
//}

void WriteImage2Jpg(unsigned char* src, int width, int height, char* filename)
{
	int y;

	Mat src_gray(height, width, CV_8U);

	for (y = 0; y < src_gray.rows; y++)
		memcpy(src_gray.data + y*src_gray.cols, src + y*src_gray.cols, src_gray.cols);

	imwrite(filename, src_gray);
}




/*
int range(int x, int y, int width, int height)
{
	if ((x < 0) || (x >= width)) {
		return(0);
	}
	if ((y < 0) || (y >= height)) {
		return(0);
	}
	return(1);
}
*/