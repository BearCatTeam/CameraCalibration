////////////////////////////////////////////////////////////
/// This program is aim to calibrate a camera with given ///
///	photos using Zhang's method                          ///
///                                                      ///
/// Author: Nguyen Hong Quan                             ///
///         Tran Nguyen Phuong Trinh                     ///
/// Email: nguyenhongquan_eeit13@hotmail.com             ///
///		   trinhtran2151995@gmail.com                    ///
/// Date: 19 April, 2016                                 ///
/// Version: 0.0.1                                       ///
////////////////////////////////////////////////////////////
//// REMEMBER : TRY TO ENHANCE THIS PROGRAM BY USING STL ///
#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2\core.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <arrayfire.h>
using namespace std;
using namespace cv;
/////////////////////////////
#define IMG_NUM 6
#define COR_NUM 5
const double PIX2MM = 0.264583333;
const double MM2PIX = 3.779527559;
String prefix = "CR"; //choose camera to calibrate. CR for cam right, CL for cam left
const int choosen_corners[COR_NUM] = { 0, 10, 6, 48, 42};  // clockwise and polygon
typedef vector<Point2f> Corners;
/////////////////////////////
vector<double> Create_v(vector<double> h, int i, int j);
//void check(vector<double> hn);
////////////////////////////
int main(int argc, char* argv[])
{
	int device = argc > 1 ? atoi(argv[1]) : 0;
	af::setDevice(device);
	af::info();
	//read input images
	String imageName;
	vector<Mat> _images;
	for (int i = 1; i <= IMG_NUM; i++)
	{
		imageName = prefix + "/Picture " + to_string(i) + ".jpg";
		_images.push_back(imread(imageName, IMREAD_GRAYSCALE));
		imageName = "";
	}
	cout << "Input Images Completed"<<endl;
	/////////////////////////////////
	// Find Chessboard Corners for every images
	Size boardSize(7, 7); // 7 corners x 7corners
	vector<Corners> _multiImageCorner; // corners vcetor of multi images
	Corners _imageCorner; // corners of 1 image 
	for (int i = 0; i < IMG_NUM; i++)
	{
		//findChessboardCorners accept vector<> for storing corners 
		bool founded = findChessboardCorners(_images[i], boardSize, _imageCorner, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (founded)
		{
			_multiImageCorner.push_back(_imageCorner);
		}
	}
	cout << "Find Chessboard Corners completed" <<endl;
	//////////////////////////////////
	// now we initialize aTxi and aTyi with i:1->5 for 5 corners pair of each images and real chessboard
	// and we will want to find H for each image => we create a loop from n:0->5 to find H for all of 6 images
	//////////////////////////////////
	vector<double> _hn[6],b; //hn of six pictures, b of Vb=0
	Mat V, P, w, vt, u; // A constructed from aTx,, aTy; V constructed from v
	for (int n = 0; n < IMG_NUM; n++) // image n 
	{
		vector<double> aTx, aTy;
		double offset = 20.00f; // the division unit of the chessboard in mm
		////////////////////
		// initialize aTx and aTy with 5 founded point of each image
		////////////////////
		for (int i = 0; i < COR_NUM; i++) //for each i corners we have a pair of aTx and aTy
		{
			aTx.clear(); aTy.clear();
			///////////////////
			aTx.push_back(-offset*(choosen_corners[i]-(choosen_corners[i]/7)*7));	  //-Xi
			aTx.push_back(-offset*(choosen_corners[i]/7));		 //-Yi
			aTx.push_back(-1.0f);
			aTx.push_back(0.0f);
			aTx.push_back(0.0f);
			aTx.push_back(0.0f);
			aTx.push_back(offset*(choosen_corners[i]-(choosen_corners[i]/7)*7)*_multiImageCorner[n][choosen_corners[i]].x*PIX2MM); //Xi*x 
			aTx.push_back(offset*(choosen_corners[i]/7)*_multiImageCorner[n][choosen_corners[i]].x*PIX2MM);//Yi*x
			aTx.push_back(_multiImageCorner[n][choosen_corners[i]].x*PIX2MM); //x
			///////////////////
			aTy.push_back(0.0f);
			aTy.push_back(0.0f);
			aTy.push_back(0.0f);
			aTy.push_back(-offset*(choosen_corners[i]-(choosen_corners[i]/7)*7)); //-Xi
			aTy.push_back(-offset*(choosen_corners[i]/7)); //-Yi
			aTy.push_back(-1.0f);
			aTy.push_back(offset*(choosen_corners[i]-(choosen_corners[i]/7)*7)*_multiImageCorner[n][choosen_corners[i]].y*PIX2MM); //-Xi*y
			aTy.push_back(offset*(choosen_corners[i]/7)*_multiImageCorner[n][choosen_corners[i]].y*PIX2MM); //-Yi*y
			aTy.push_back(_multiImageCorner[n][choosen_corners[i]].y*PIX2MM);	//y
			///////////////////
			Mat aTx_tmp(aTx, true);
			aTx_tmp = aTx_tmp.t();
			Mat aTy_tmp(aTy, true);
			aTy_tmp = aTy_tmp.t();
			P.push_back(aTx_tmp);
			P.push_back(aTy_tmp);
			aTx_tmp.release();
			aTy_tmp.release();
		}
		//we have here A is M which comprise of aTx.h =0 and aTy.h = 0
		//cout << endl <<"Print A" << endl << A << endl << endl;
		// w is eigenvalue of hT.h => w[8] is the smallest eigenvalue
		// vt is eigenvector of hT.h
		double *P2 = new double(P.cols*P.rows);
		P2 = (double*)P.data;
		af::array P3(P.cols,P.rows,P2,afHost); P3=P3.T();
		af::array ua, sa, vta;
		af::svd(ua, sa, vta, P3);
		cout << "With Array Fire: " << endl;
		af_print(ua);
		cout << endl;
		af_print(sa);
		cout << endl;
		af_print(vta);
		cout << endl;
		cout << "With OpenCV: " << endl;
		SVD::compute(P,w,u,vt,SVD::FULL_UV);
		//vt = vt.t(); //SVD will return Vt so we need to transpose back to V
		cout <<"Print VT"<<endl<<vt<<endl<<endl;
		cout << "Print U" << endl << u << endl << endl;
		cout << "Print W" << endl << w << endl << endl;
		_hn[n] = vt.col(8); // choosing the eigenvector in assosiate with smallest eigenvalue w[8];
		//Mat hn(_hn[n], true); => just use for printing out _hn[n]
		//cout <<"Print h"<<endl<<hn<<endl<<endl;
		//check(_hn[n]); // this is just a function to check wether compute h is appropriate or not
		P.release(); vt.release(); u.release(); w.release();
		//////////everything upto here is quite ok/////////////


		vector<double> v12 = Create_v(_hn[n], 1, 2);
		vector<double> v11 = Create_v(_hn[n], 1, 1);
		vector<double> v22 = Create_v(_hn[n], 2, 2);
		vector<double> v(6); // v = v11-v22
		Mat v12_tmp(v12, true); 
		v12_tmp = v12_tmp.t();
		V.push_back(v12_tmp);
		v12_tmp.release(); v12.clear();
		
		std::transform(v11.begin(), v11.end(), v22.begin(), v.begin(), std::minus<double>());
		Mat v_tmp(v, true);
		v_tmp = v_tmp.t();
		V.push_back(v_tmp);
		v_tmp.release();  v11.clear(); v22.clear(); v.clear();
	}
	_hn->clear();
	cout << endl <<"Print V" << endl << V << endl << endl;
	SVD::compute(V, w, u, vt, SVD::FULL_UV);
	vt = vt.t();
	cout <<"Print VT"<<endl<<vt<<endl<<endl;
	//cout << "Print U" << endl << u << endl << endl;
	cout << "Print W" << endl << w << endl << endl;
	b = vt.col(5); // b = [B11,B12;B13;B22;B23;B33]
	cout << endl << "Print b: " << endl;
	for_each(b.begin(), b.end(), [](double x){cout << x << " "; });
	cout << endl;
	////////////////////
	//// calculate A with B=At*A
	Mat A(3,3,CV_64F,0.0f);
	A.at<double>(0, 0) = sqrt(abs(b[0]));
	A.at<double>(0, 1) = b[1]/(A.at<double>(0,0));
	A.at<double>(0, 2) = b[2]/(A.at<double>(0,0));
	A.at<double>(1, 1) = sqrt(abs(b[3]-pow(A.at<double>(0,1),2)));
	A.at<double>(1, 2) = (b[4]-A.at<double>(0,1)*A.at<double>(0,2))/(A.at<double>(1,1));
	A.at<double>(2, 2) = sqrt(abs(b[5]-pow(A.at<double>(0,2),2)-pow(A.at<double>(1,2),2)));
	cout << endl << "Print A: " << endl << A << endl;
	Mat preK = A.inv(DECOMP_LU);
	cout << endl << "Print Preprocessed K: " << endl << preK << endl;
	return 0;
}

/*This function is used to create v small of V big*/
vector<double> Create_v(const vector<double> h, int i, int j)
{
	vector<double> result;
	int h1x = -1;
	int h2x = 2;
	int h3x = 5;
	result.push_back(h[h1x+i]*h[h1x+j]);
	result.push_back(h[h1x+i]*h[h2x+j]+h[h2x+i]*h[h1x+j]);
	result.push_back(h[h3x+i]*h[h1x+j]+h[h1x+i]*h[h3x+j]);
	result.push_back(h[h2x+i]*h[h2x+j]);
	result.push_back(h[h3x+i]*h[h2x+j]+h[h2x+i]*h[h3x+j]);
	result.push_back(h[h3x+i]*h[h3x+j]);
	return result;
}

/*void check(vector<double> hn)
{
	Mat H,x;
	vector<double> tmp_vect;
	double xcoor, ycoor;
	cout << endl << "Enter 3D xcoor (mm): ";
	cin >> xcoor;
	cout << endl << "Enter 3D ycoor (mm): ";
	cin >> ycoor;
	for (int i = 0; i < hn.size(); i+=3)
	{
		for (int j = i; j <= i+2; j++)
		{
			tmp_vect.push_back(hn[j]);
		}
		Mat tmp_mat(tmp_vect, true); tmp_mat = tmp_mat.t();
		H.push_back(tmp_mat);
		tmp_vect.clear(); tmp_mat.release();
	}
	tmp_vect.push_back(xcoor); tmp_vect.push_back(ycoor); tmp_vect.push_back(1);
	cout << endl << "H is" << endl << H << endl;
	Mat X(tmp_vect, true);
	x = H*X; // this will solve for x in homography geometry x=[u;v;w] 
	// to have x in euclidian (x=[x;y;1]) ; x=u/w ; y=v/w
	x.at<double>(0, 0) = (x.at<double>(0, 0) / x.at<double>(2, 0))*MM2PIX;
	x.at<double>(1, 0) = (x.at<double>(1, 0) / x.at<double>(2, 0))*MM2PIX;
	x.at<double>(2, 0) = x.at<double>(2, 0) / x.at<double>(2, 0);
	cout <<endl<<"Point is (pixel):" <<endl<<x<< endl;
	system("pause");		  
	}	 */