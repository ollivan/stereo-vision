// homework_vision.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp> 
#include <cmath>
#include<time.h>


using namespace std;
using namespace cv;

const int imageWidth = 2048;                             //摄像头的分辨率  
const int imageHeight = 1536;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;  		//校正旋转矩阵R，投影矩阵P 重投影矩阵Q
double AR=0,BR=0, CR=0, FR=0, OLx = 0, ORy = 0;            
Mat xyz;              //三维坐标
vector<Point2f> pointsA1L(4);
vector<Point2f> pointsA1R(4);
vector<Point3f> pointsA13d(4);
Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
Point2f R5(1581,939),R6(1587,1038),L5(1086,933),L6(1089,1026);
/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 3517.2435, 0, 1026.7696,
	0, 3516.7378, 763.1432,
	0, 0, 1);
//对应matlab里的左相机标定矩阵
Mat distCoeffL = (Mat_<double>(5, 1) << -0.0700, -0.2442, -0.0001, -0.0012, 8.3048);
//对应Matlab所得左相机畸变参数

Mat cameraMatrixR = (Mat_<double>(3, 3) << 3517.6538, 0, 1044.3345,
	0, 3518.0360, 746.2314,
	0, 0, 1);
//对应matlab里的右相机标定矩阵

Mat distCoeffR = (Mat_<double>(5, 1) << -0.0505, -0.9744, 0.0010, -0.0009, 16.5803);
//对应Matlab所得右相机畸变参数

Mat T = (Mat_<double>(3, 1) << 102.0327, 3.5357, 10.0266);//T平移向量
													//对应Matlab所得T参数
//Mat rec = (Mat_<double>(3, 1) << 0.15293, 0.12915, -0.01944);//rec旋转向量，对应matlab om参数
Mat R = (Mat_<double>(3, 3) << 0.9817, 0.0793, -0.1731, 
	-0.0817 , 0.9966 , -0.0066,
	0.1720, 0.0206, 0.9849);//R 旋转矩阵



/*****计算椭圆参数*****/

struct EllipsePara
{
	Point2f c,a1,a2,b1,b2;
	float A;
	float B;
	float C;
	float F;
	float R;
};
void getEllipsePara(RotatedRect & ellipsemege, EllipsePara& EP_t)
{
	
 
	float theta = ellipsemege.angle * CV_PI / 180.0 ;
	float a = ellipsemege.size.width / 2.0;
	float b = ellipsemege.size.height / 2.0;
 	EP_t.R = (a+b)/2.0;
	EP_t.c.x = ellipsemege.center.x;
	EP_t.c.y = ellipsemege.center.y;
 
	EP_t.a1.x = ellipsemege.center.x + a * cos(theta) ;
	EP_t.a1.y = ellipsemege.center.y + a * sin(theta) ;
	
	EP_t.a2.x = ellipsemege.center.x - a * cos(theta) ;
	EP_t.a2.y = ellipsemege.center.y - a * sin(theta) ;
	
	EP_t.b1.x = ellipsemege.center.x - b * sin(theta) ;
	EP_t.b1.y = ellipsemege.center.y + b * cos(theta) ;
	
	EP_t.b2.x = ellipsemege.center.x + b * sin(theta) ;
	EP_t.b2.y = ellipsemege.center.y - b * cos(theta) ;

	EP_t.A = a * a * sin(theta) * sin(theta) + b * b * cos(theta) * cos(theta);
	EP_t.B = (-2.0) * (a * a - b * b) * sin(theta) * cos(theta);
	EP_t.C = a * a * cos(theta) * cos(theta) + b * b * sin(theta) * sin(theta);
	EP_t.F = (-1.0) * a * a * b * b;
 
 
	//cout << "theta: " << theta << " x: " << EP.c.x << " y: " << EP.c.y << " C: " << EP.C << " F: " << EP.F << endl;
 
 
}
/*******在右图中搜索匹配点***********/
double matchR2L(double xl,double yl)
{	
	yl = yl -ORy;
	double xr = 0;
	double b = BR*yl;
	double a = AR;
 	double c = CR*yl*yl + FR;
	if(b*b - 4*a*c < 0) {return 0;}
	if(xl > OLx)
	xr = (-b + sqrt(b*b - 4*a*c))/(2*a);
	else 	
	xr = (-b - sqrt(b*b - 4*a*c))/(2*a);
	return xr;
}
/*****主函数*****/
int main()
{
	/*立体校正*/
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

	/*读取图片*/
	rgbImageL = imread("left197.bmp", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImageL, grayImageL, COLOR_RGB2GRAY);
	rgbImageR = imread("right197.bmp", CV_LOAD_IMAGE_COLOR);
	cvtColor(rgbImageR, grayImageR, COLOR_RGB2GRAY);
	//imshow("ImageL Before Rectify", grayImageL);
	//imshow("ImageR Before Rectify", grayImageR);

	/*
	经过remap之后，左右相机的图像已经共面并且行对准了
	*/
	remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
	remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

	/*
	把校正结果显示出来
	*/
	Mat rgbRectifyImageL, rgbRectifyImageR;
	cvtColor(rectifyImageL, rgbRectifyImageL, COLOR_GRAY2BGR);  //伪彩色图
	cvtColor(rectifyImageR, rgbRectifyImageR, COLOR_GRAY2BGR);

	//单独显示
//	rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
//	rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
	//imshow("ImageL After Rectify", rgbRectifyImageL);
	//imshow("ImageR After Rectify", rgbRectifyImageR);
	imwrite("ImageLAfterRectify.bmp", rgbRectifyImageL);
	imwrite("ImageRAfterRectify.bmp", rgbRectifyImageR);

	vector<vector<Point>> contoursImageL, contoursImageR;
	vector<Vec4i> hierarchyL,hierarchyR;
	rectifyImageL = rectifyImageL > 100;
	rectifyImageR = rectifyImageR > 100;
	findContours(rectifyImageL, contoursImageL,hierarchyL, CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	findContours(rectifyImageR, contoursImageR, hierarchyR,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE);
	Mat dstImageL, dstImageR;
	dstImageL = Mat::zeros(rectifyImageL.size(), CV_8UC3);
	dstImageR = Mat::zeros(rectifyImageR.size(), CV_8UC3);
	RotatedRect boxL,boxR;
	EllipsePara ellipPaL,ellipPaR;
	
	for(int i = 0; i< hierarchyL.size(); i++)	
	{
		if(contoursImageL.at(i).size() < 500 && contoursImageL.at(i).size() > 350)
		{
			RotatedRect boxL = fitEllipse(contoursImageL[i]);
			if( MAX(boxL.size.width, boxL.size.height) < MIN(boxL.size.width, boxL.size.height)*1.5 )
			{
				drawContours(dstImageL,contoursImageL,i,Scalar(0, 0, 255),CV_FILLED,8,hierarchyL);
				ellipse(rgbRectifyImageL, boxL, Scalar(0,255,0), 1, CV_AA);
				getEllipsePara(boxL,ellipPaL);
			}	
		}			
	}
	OLx = ellipPaL.c.x;
//	imshow("ImageLContour.bmp",rgbRectifyImageL);
	for(int i = 0; i< hierarchyR.size(); i++)	
	{
		if(contoursImageR.at(i).size() > 500 || contoursImageR.at(i).size() < 165)
			continue;
		RotatedRect boxR = fitEllipse(contoursImageR[i]);
		if( MAX(boxR.size.width, boxR.size.height) > MIN(boxR.size.width, boxR.size.height)*1.5 )
			continue;
		drawContours(dstImageR,contoursImageR,i,Scalar(0, 0, 255),CV_FILLED,8,hierarchyR);
		ellipse(rgbRectifyImageR, boxR, Scalar(0,255,0), 1, CV_AA);
	
		getEllipsePara(boxR,ellipPaR);		
		AR=ellipPaR.A;BR=ellipPaR.B;CR=ellipPaR.C;FR=ellipPaR.F;ORy = ellipPaR.c.y;
		
		break;
												
	}
	FileStorage point;
	time_t rawtime;time(&rawtime);
	Point2f L1(ellipPaL.a1.x,ellipPaL.a1.y);
	Point2f R1(matchR2L(ellipPaL.a1.x,ellipPaL.a1.y) + ellipPaR.c.x,ellipPaL.a1.y -ellipPaL.c.y + ellipPaR.c.y);
	Point2f L2(ellipPaL.a2.x,ellipPaL.a2.y);
	Point2f R2(matchR2L(ellipPaL.a2.x,ellipPaL.a2.y) + ellipPaR.c.x,ellipPaL.a2.y -ellipPaL.c.y + ellipPaR.c.y);
	Point2f L3(ellipPaL.b1.x,ellipPaL.b1.y);
	Point2f R3(matchR2L(ellipPaL.b1.x,ellipPaL.b1.y) + ellipPaR.c.x,ellipPaL.b1.y -ellipPaL.c.y + ellipPaR.c.y);
	Point2f L4(ellipPaL.b2.x,ellipPaL.b2.y);
	Point2f R4(matchR2L(ellipPaL.b2.x,ellipPaL.b2.y) + ellipPaR.c.x,ellipPaL.b2.y -ellipPaL.c.y + ellipPaR.c.y);
	point.open("point.yml",FileStorage::WRITE);
	point<<"pointLc"<<ellipPaR.c;
	point<<"pointL1"<<L1;
	point<<"pointR1"<<R1;
	point<<"pointL2"<<L2;
	point<<"pointR2"<<R2;
	point<<"pointL3"<<L3;
	point<<"pointR3"<<R3;
	point<<"pointL4"<<L4;
	point<<"pointR4"<<R4;
	cout<<"/****************************************************************/"<<endl;
	cout<<"The coordinates of match points have been written into 'point.yml'"<<endl;
	cout<<"/****************************************************************/"<<endl;
	//cout<<matchR2L(ellipPaL.a1.x,ellipPaL.a1.y) + ellipPaR.c.x<<'\t'<<ellipPaL.a1.y -ellipPaL.c.y + ellipPaR.c.y<<endl;
	//cout<<matchR2L(ellipPaL.a2.x,ellipPaL.a2.y) + ellipPaR.c.x<<'\t'<<ellipPaL.a2.y-ellipPaL.c.y + ellipPaR.c.y<<endl;
	//cout<<matchR2L(ellipPaL.b1.x,ellipPaL.b1.y) + ellipPaR.c.x<<'\t'<<ellipPaL.b1.y-ellipPaL.c.y + ellipPaR.c.y<<endl;
	//cout<<matchR2L(ellipPaL.b2.x,ellipPaL.b2.y) + ellipPaR.c.x<<'\t'<<ellipPaL.b2.y-ellipPaL.c.y + ellipPaR.c.y<<endl;
//	imshow("ImageRContour.bmp",rgbRectifyImageR);
	

/*
	Mat disp;
	int mindisparity = -1;
	int ndisparities = 64;  
	int SADWindowSize = 5; 
	//SGBM
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * rgbRectifyImageL.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * rgbRectifyImageL.channels() * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);
	sgbm->compute(rgbRectifyImageL, rgbRectifyImageR, disp);
	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
	Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
	normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	//imshow("",disp8U);	
	//imwrite("results/SGBM.jpg", disp8U);
	reprojectImageTo3D(disp, xyz, Q, true);
	xyz =16*xyz;
	for(double i = MIN (ellipPaL.a1.x,ellipPaL.a2.x); 
		i<MAX (ellipPaL.a1.x,ellipPaL.a2.x);i=i+0.1)
	{
		for(double j = MIN (ellipPaL.b1.y,ellipPaL.b2.y) - 100.00; 
		j < MAX (ellipPaL.b1.y,ellipPaL.b2.y) + 100.0;j=j+0.1)
		{
			if(abs(ellipPaL.A*(i-ellipPaL.c.x)*(i-ellipPaL.c.x)+ellipPaL.B*(j-ellipPaL.c.y)*(j-ellipPaL.c.y)+ellipPaL.C*(j-ellipPaL.c.y)*(i-ellipPaL.c.x)+ellipPaL.F) < 0.005)
			cout<<i<<","<<j<<"the 3d coordinate is "<<xyz.at<Vec3f>(i,j)<<endl;
		}
	}
	
	cout<<"the 3d coordinate is "<<xyz.at<Vec3f>(ellipPaL.a1.x,ellipPaL.a1.y)<<'\t'
	<<xyz.at<Vec3f>(ellipPaL.a2.x,ellipPaL.a2.y)<<'\t'
	<<xyz.at<Vec3f>(ellipPaL.b1.x,ellipPaL.b1.y)<<'\t'
	<<xyz.at<Vec3f>(ellipPaL.b2.x,ellipPaL.b2.y)<<endl;
	cout<<xyz.at<Vec3f>(ellipPaL.c.x,ellipPaL.c.y+ellipPaL.R)<<endl;
	cout<<xyz.at<Vec3f>(ellipPaL.c.x,ellipPaL.c.y-ellipPaL.R)<<endl;
	
	
	/*
	for(int i = 0; i< hierarchyL.size() ;i++)
	{
		Scalar color = Scalar(rand()%255 ,rand()%255 ,rand()%255);
		drawContours(dstImageL,contoursImageL,i,color,CV_FILLED,8,hierarchyL);	
	}
	for(int j = 0; j< hierarchyR.size() ;j++)
	{
		Scalar color = Scalar(rand()%255 ,rand()%255 ,rand()%255);
		drawContours(dstImageR,contoursImageR,j,color,CV_FILLED,8,hierarchyR);	
	}
	*/
	//imshow("ImageLContour",dstImageL);
	//imshow("ImageRContour",dstImageR);	
	//imwrite("ImageLContour.bmp",dstImageL);
	//imwrite("ImageRContour.bmp",dstImageR);
	

	point<<"pointL5"<<L5;
	point<<"pointR5"<<R5;
	point<<"pointL6"<<L6;
	point<<"pointR6"<<R6;
	point<<"Time"<<asctime(localtime(&rawtime));
	point.release();
/*
	//显示在同一张图上
	Mat canvas;
	double sf;
	int w, h;
	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width * sf);
	h = cvRound(imageSize.height * sf);
	canvas.create(h, w * 2, CV_8UC3);   //注意通道

										//左图像画到画布上
	Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
	resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
	Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
		cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
	//rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
	cout << "Painted ImageL" << endl;

	//右图像画到画布上
	canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
	resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
	Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
		cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
	//rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
	cout << "Painted ImageR" << endl;

	//画上对应的线条
	for (int i = 0; i < canvas.rows; i += 16)
		line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
//	imshow("rectified", canvas);
	int k = 1;	
	string Imagename = "Rectify"+ to_string(k)+".bmp";
	imshow("Rectify"+ to_string(k) , canvas);	
	imwrite(Imagename, canvas);

/*
	
	
*/
	
	waitKey(0);
	return 0;
}
