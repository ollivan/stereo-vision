#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <cmath>
 
using namespace std;
using namespace cv;
 
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3]);
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight);
bool sphereLeastFit(const std::vector<Point3f> &points,
                    double &center_x, double &center_y, double &center_z, double &radius);
int cagaus(double a[],double b[],int n,double x[]);
 
//左相机内参数矩阵
float leftIntrinsic[3][3] = {3517.2435, 0, 1026.7696,
	0, 3516.7378, 763.1432,
	0, 0, 1};
//左相机畸变系数
float leftDistortion[1][5] = {-0.0700, -0.2442, -0.0001, -0.0012, 8.3048};
//左相机旋转矩阵
float leftRotation[3][3] = {1,0,0, 0,1,0, 0,0,1};
Mat Rt = (Mat_<double>(3,4)<<0.9817, 0.0793, -0.1731, 102.0327,
	-0.0817 , 0.9966 , -0.0066 ,3.5357,
	0.1720, 0.0206, 0.9849 ,10.0266);
//左相机平移向量
float leftTranslation[1][3] = {0,0,0};
 
//右相机内参数矩阵
float rightIntrinsic[3][3] = {3517.6538, 0, 1044.3345,
	0, 3518.0360, 746.2314,
	0, 0, 1};
//右相机畸变系数
float rightDistortion[1][5] = {-0.0505, -0.9744, 0.0010, -0.0009, 16.5803};
//右相机旋转矩阵
float rightRotation[3][3] = { 0.9817, 0.0793, -0.1731, 
	-0.0817 , 0.9966 , -0.0066,
	0.1720, 0.0206, 0.9849};
//右相机平移向量
float rightTranslation[1][3] = {131.1852, 4.4787, 12.8074};
 vector<Point2f> input(1);
 vector<Point3f> output(1);
 
int main()
{
	FileStorage point("point.yml",FileStorage::READ);
	
	Point2f l1,l2,l3,l4,l5,l6;
	Point2f r1,r2,r3,r4,r5,r6;
	Point3f worldPoint1,worldPoint2,worldPoint3,worldPoint4,worldPoint5,worldPoint6;
	worldPoint1 = uv2xyz(l4,r4);
	worldPoint2 = uv2xyz(l5,r5);
	point["pointL1"]>>l1;
	point["pointR1"]>>r1;
	point["pointL2"]>>l2;
	point["pointR2"]>>r2;
	point["pointL3"]>>l3;
	point["pointR3"]>>r3;
	point["pointL4"]>>l4;
	point["pointR4"]>>r4;	
	point["pointL5"]>>l5;
	point["pointR5"]>>r5;
	point["pointL6"]>>l6;
	point["pointR6"]>>r6;
	worldPoint1 = uv2xyz(l1,r1);
	worldPoint2 = uv2xyz(l2,r2);
	worldPoint3 = uv2xyz(l3,r3);
	worldPoint4 = uv2xyz(l4,r4);
	worldPoint5 = uv2xyz(l5,r5);	
	worldPoint6 = uv2xyz(l6,r6);
	vector<Point3f> match = {worldPoint1,worldPoint3,worldPoint4,worldPoint5,worldPoint6};
	double d = sqrt( pow((worldPoint5.x - worldPoint6.x),2) + pow((worldPoint5.y - worldPoint6.y),2) + pow((worldPoint1.z - worldPoint2.z),2)); 
	double x,y,z,r;
	cout<<worldPoint1<<endl;	
	cout<<worldPoint2<<endl;
	cout<<worldPoint3<<endl;	
	cout<<worldPoint4<<endl;	
	cout<<worldPoint5<<endl;	
	cout<<worldPoint6<<endl;		
	if(sphereLeastFit(match,x,y,z,r));
	
	cout<<r<<endl;
 
 
	return 0;
}
 
 
//根据相机标定结果求三维坐标
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight)
{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat mLeftRotation = Mat(3,3,CV_32F,leftRotation);
	Mat mLeftTranslation = Mat(3,1,CV_32F,leftTranslation);
	Mat mLeftRT = Mat(3,4,CV_32F);//左相机M矩阵
	hconcat(mLeftRotation,mLeftTranslation,mLeftRT);
	Mat mLeftIntrinsic = Mat(3,3,CV_32F,leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;
 
	Mat mRightRotation = Mat(3,3,CV_32F,rightRotation);
	Mat mRightTranslation = Mat(3,1,CV_32F,rightTranslation);
	Mat mRightRT = Mat(3,4,CV_32F);//右相机M矩阵
	hconcat(mRightRotation,mRightTranslation,mRightRT);
	Mat mRightIntrinsic = Mat(3,3,CV_32F,rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;
 
	//最小二乘法A矩阵
	Mat A = Mat(4,3,CV_32F);
	A.at<float>(0,0) = uvLeft.x * mLeftM.at<float>(2,0) - mLeftM.at<float>(0,0);
	A.at<float>(0,1) = uvLeft.x * mLeftM.at<float>(2,1) - mLeftM.at<float>(0,1);
	A.at<float>(0,2) = uvLeft.x * mLeftM.at<float>(2,2) - mLeftM.at<float>(0,2);
 
	A.at<float>(1,0) = uvLeft.y * mLeftM.at<float>(2,0) - mLeftM.at<float>(1,0);
	A.at<float>(1,1) = uvLeft.y * mLeftM.at<float>(2,1) - mLeftM.at<float>(1,1);
	A.at<float>(1,2) = uvLeft.y * mLeftM.at<float>(2,2) - mLeftM.at<float>(1,2);
 
	A.at<float>(2,0) = uvRight.x * mRightM.at<float>(2,0) - mRightM.at<float>(0,0);
	A.at<float>(2,1) = uvRight.x * mRightM.at<float>(2,1) - mRightM.at<float>(0,1);
	A.at<float>(2,2) = uvRight.x * mRightM.at<float>(2,2) - mRightM.at<float>(0,2);
 
	A.at<float>(3,0) = uvRight.y * mRightM.at<float>(2,0) - mRightM.at<float>(1,0);
	A.at<float>(3,1) = uvRight.y * mRightM.at<float>(2,1) - mRightM.at<float>(1,1);
	A.at<float>(3,2) = uvRight.y * mRightM.at<float>(2,2) - mRightM.at<float>(1,2);
 
	//最小二乘法B矩阵
	Mat B = Mat(4,1,CV_32F);
	B.at<float>(0,0) = mLeftM.at<float>(0,3) - uvLeft.x * mLeftM.at<float>(2,3);
	B.at<float>(1,0) = mLeftM.at<float>(1,3) - uvLeft.y * mLeftM.at<float>(2,3);
	B.at<float>(2,0) = mRightM.at<float>(0,3) - uvRight.x * mRightM.at<float>(2,3);
	B.at<float>(3,0) = mRightM.at<float>(1,3) - uvRight.y * mRightM.at<float>(2,3);
 
	Mat XYZ = Mat(3,1,CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A,B,XYZ,DECOMP_SVD);
 
	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;
 
	//世界坐标系中坐标
	Point3f world;
	world.x = XYZ.at<float>(0,0);
	world.y = XYZ.at<float>(1,0);
	world.z = XYZ.at<float>(2,0);
 
	return world;
}
 

//三维坐标转二维坐标
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3])
{
	//    [fx s x0]							[Xc]		[Xw]		[u]	  1		[Xc]
	//K = |0 fy y0|       TEMP = [R T]		|Yc| = TEMP*|Yw|		| | = —*K *|Yc|
	//    [ 0 0 1 ]							[Zc]		|Zw|		[v]	  Zc	[Zc]
	//													[1 ]
	Point3f c;
	c.x = rotation[0][0]*worldPoint.x + rotation[0][1]*worldPoint.y + rotation[0][2]*worldPoint.z + translation[0][0]*1;
	c.y = rotation[1][0]*worldPoint.x + rotation[1][1]*worldPoint.y + rotation[1][2]*worldPoint.z + translation[0][1]*1;
	c.z = rotation[2][0]*worldPoint.x + rotation[2][1]*worldPoint.y + rotation[2][2]*worldPoint.z + translation[0][2]*1;
 
	Point2f uv;
	uv.x = (intrinsic[0][0]*c.x + intrinsic[0][1]*c.y + intrinsic[0][2]*c.z)/c.z;
	uv.y = (intrinsic[1][0]*c.x + intrinsic[1][1]*c.y + intrinsic[1][2]*c.z)/c.z;
 
	return uv;
}
//拟合球面
// 全选主元高斯消去法
//a-n*n 存放方程组的系数矩阵，返回时将被破坏
//b-常数向量
//x-返回方程组的解向量
//n-存放方程组的阶数
//返回0表示原方程组的系数矩阵奇异
int cagaus(double a[],double b[],int n,double x[])
{
    int *js, l, k, i, j, is, p, q;
    double d, t;
    js=new int [n];
    l=1;
    for (k=0;k<=n-2;k++)
    {
        d=0.0;
        for (i=k;i<=n-1;i++)
        {
            for (j=k;j<=n-1;j++)
            {
                t=fabs(a[i*n+j]);
                if (t>d)
                {
                    d=t;
                    js[k]=j;
                    is=i;
                }
            }
        }
        if (d+1.0==1.0)
        {
            l=0;
        }
        else
        {
            if (js[k]!=k)
            {
                for (i=0;i<=n-1;i++)
                {
                    p=i*n+k;
                    q=i*n+js[k];
                    t=a[p];
                    a[p]=a[q];
                    a[q]=t;
                }
            }
            if (is!=k)
            {
                for (j=k;j<=n-1;j++)
                {
                    p=k*n+j;
                    q=is*n+j;
                    t=a[p];
                    a[p]=a[q];
                    a[q]=t;
                }
                t=b[k];
                b[k]=b[is];
                b[is]=t;
            }
        }
        if (l==0)
        {
            delete js;
            return(0);
        }
        d=a[k*n+k];
        for (j=k+1;j<=n-1;j++)
        {
            p=k*n+j;
            a[p]=a[p]/d;
        }
        b[k]=b[k]/d;
        for (i=k+1;i<=n-1;i++)
        {
            for (j=k+1;j<=n-1;j++)
            {
                p=i*n+j;
                a[p]=a[p]-a[i*n+k]*a[k*n+j];
            }
            b[i]=b[i]-a[i*n+k]*b[k];
        }
    }
    d=a[(n-1)*n+n-1];
    if (fabs(d)+1.0==1.0)
    {
        delete js;
        return(0);
    }
    x[n-1]=b[n-1]/d;
    for (i=n-2;i>=0;i--)
    {
        t=0.0;
        for (j=i+1;j<=n-1;j++)
        {
            t=t+a[i*n+j]*x[j];
        }
        x[i]=b[i]-t;
    }
    js[n-1]=n-1;
    for (k=n-1;k>=0;k--)
    {
        if (js[k]!=k)
        {
            t=x[k];
            x[k]=x[js[k]];
            x[js[k]]=t;
        }
    }
    delete js;
    return 1;
}
bool sphereLeastFit(const std::vector<Point3f> &points,
                    double &center_x, double &center_y, double &center_z, double &radius)
{
    center_x = 0.0f;
    center_y = 0.0f;
    radius = 0.0f;

    int N = points.size();
    if (N < 4) // 最少 4 个点才能拟合出一个球面。
    {
        return false;
    }

    double sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;

    for (int i = 0; i < N; i++)
    {
        double x = points[i].x;
        double y = points[i].y;
        double z = points[i].z;
        sum_x += x;
        sum_y += y;
        sum_z += z;
    }
    double mean_x = sum_x / N;
    double mean_y = sum_y / N;
    double mean_z = sum_z / N;
    double sum = 0.0;
    double sum_u2 = 0.0f, sum_v2 = 0.0f, sum_w2 = 0.0f;
    double sum_u3 = 0.0f, sum_v3 = 0.0f, sum_w3 = 0.0f;
    double sum_uv = 0.0f, sum_uw = 0.0f, sum_vw = 0.0f;
    double sum_u2v = 0.0f, sum_uv2 = 0.0f, sum_u2w = 0.0f, sum_v2w = 0.0f, sum_uw2 = 0.0f, sum_vw2 = 0.0f;
    for (int i = 0; i < N; i++)
    {
        double u = points[i].x - mean_x;
        double v = points[i].y - mean_y;
        double w = points[i].z - mean_z;
        double u2 = u * u;
        double v2 = v * v;
        double w2 = w * w;
        sum_u2 += u2;
        sum_v2 += v2;
        sum_w2 += w2;
        sum_u3 += u2 * u;
        sum_v3 += v2 * v;
        sum_w3 += w2 * w;
        sum_uv += u * v;
        sum_vw += v * w;
        sum_uw += u * w;
        sum_u2v += u2 * v;
        sum_u2w += u2 * w;
        sum_uv2 += u * v2;
        sum_v2w += v2 * w;
        sum_uw2 += u * w2;
        sum_vw2 += v * w2;
    }

    double A[3][3];
    double B[3] = {(sum_u3 + sum_uv2 + sum_uw2) / 2.0,
                   (sum_u2v + sum_v3 + sum_vw2) / 2.0,
                   (sum_u2w + sum_v2w + sum_w3) / 2.0};

    A[0][0] = sum_u2;
    A[0][1] = sum_uv;
    A[0][2] = sum_uw;
    A[1][0] = sum_uv;
    A[1][1] = sum_v2;
    A[1][2] = sum_vw;
    A[2][0] = sum_uw;
    A[2][1] = sum_vw;
    A[2][2] = sum_w2;

    double ans[3];

    if( cagaus((double *) A, B, 3, ans) == 0)
    {
        return false;
    }
    else
    {
        center_x = ans[0] + mean_x;
        center_y = ans[1] + mean_y;
        center_z = ans[2] + mean_z;

 
        for(int i = 0; i < N; i++)
        {
            double dx = points[i].x - center_x;
            double dy = points[i].y - center_y;
            double dz = points[i].z - center_z;
            sum += dx * dx + dy * dy + dz * dz;
        }
        sum /= N;
        radius = sqrt(sum);
    }
    return true;
}

