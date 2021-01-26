// AlsPlaneExtract.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//Author:mao ning
//Email:980913140@qq.com
//date:2020-12-14
//message:提取机载点云屋顶面


#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/registration/icp.h>
#include<pcl/visualization/pcl_visualizer.h>//可视化头文件
#include <pcl/filters/radius_outlier_removal.h>
#include<pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include "lasreader.hpp"
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>



using namespace std;
using namespace cv;

typedef pcl::PointXYZI ptype;
typedef pcl::PointCloud<ptype>::Ptr ptrtype;

static std::string convertFilePath(const std::string& file)
{
	int i = 0;
	std::string s(file);
	for (i = 0; i < s.size(); ++i)
	{
		if (s[i] == '/')
			s[i] = '\\';
	}
	return s;
}
typedef struct {
	int grayScale = 0;
	float maxdif_height = 0;
	float region_difheight = 0;
	float max_height = 0;
	int candidate = 0;
	vector<int> indexID;

}flat_grid;
typedef struct {
	vector<int> als_index;
	vector<Vec2f> xy_drift;
}lines_combination;
typedef struct {
	pcl::PointXYZ startpoint;
	pcl::PointXYZ endpoint;
}Point_vec;



int findmaxValue(Mat img, int rol, int col)
{
	int maxV = 0;
	for (int i = rol - 1; i <= rol + 1; i++)
	{
		for (int j = col - 1; j <= col + 1; j++)
		{
			if (i < 0 || j < 0) continue;
			if (maxV < img.ptr<uchar>(i)[j]) maxV = img.ptr<uchar>(i)[j];
		}
	}
	return maxV;
}


void thinImage(Mat & srcImg) {
	vector<Point> deleteList;
	int neighbourhood[9];
	int nl = srcImg.rows;
	int nc = srcImg.cols;
	bool inOddIterations = true;
	while (true) {
		for (int j = 1; j < (nl - 1); j++) {
			uchar* data_last = srcImg.ptr<uchar>(j - 1);
			uchar* data = srcImg.ptr<uchar>(j);
			uchar* data_next = srcImg.ptr<uchar>(j + 1);
			for (int i = 1; i < (nc - 1); i++) {
				if (data[i] == 255) {
					int whitePointCount = 0;
					neighbourhood[0] = 1;
					if (data_last[i] == 255) neighbourhood[1] = 1;
					else  neighbourhood[1] = 0;
					if (data_last[i + 1] == 255) neighbourhood[2] = 1;
					else  neighbourhood[2] = 0;
					if (data[i + 1] == 255) neighbourhood[3] = 1;
					else  neighbourhood[3] = 0;
					if (data_next[i + 1] == 255) neighbourhood[4] = 1;
					else  neighbourhood[4] = 0;
					if (data_next[i] == 255) neighbourhood[5] = 1;
					else  neighbourhood[5] = 0;
					if (data_next[i - 1] == 255) neighbourhood[6] = 1;
					else  neighbourhood[6] = 0;
					if (data[i - 1] == 255) neighbourhood[7] = 1;
					else  neighbourhood[7] = 0;
					if (data_last[i - 1] == 255) neighbourhood[8] = 1;
					else  neighbourhood[8] = 0;
					for (int k = 1; k < 9; k++) {
						whitePointCount += neighbourhood[k];
					}
					if ((whitePointCount >= 2) && (whitePointCount <= 6)) {
						int ap = 0;
						if ((neighbourhood[1] == 0) && (neighbourhood[2] == 1)) ap++;
						if ((neighbourhood[2] == 0) && (neighbourhood[3] == 1)) ap++;
						if ((neighbourhood[3] == 0) && (neighbourhood[4] == 1)) ap++;
						if ((neighbourhood[4] == 0) && (neighbourhood[5] == 1)) ap++;
						if ((neighbourhood[5] == 0) && (neighbourhood[6] == 1)) ap++;
						if ((neighbourhood[6] == 0) && (neighbourhood[7] == 1)) ap++;
						if ((neighbourhood[7] == 0) && (neighbourhood[8] == 1)) ap++;
						if ((neighbourhood[8] == 0) && (neighbourhood[1] == 1)) ap++;
						if (ap == 1) {
							if (inOddIterations && (neighbourhood[3] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[5] == 0)) {
								deleteList.push_back(Point(i, j));
							}
							else if (!inOddIterations && (neighbourhood[1] * neighbourhood[5] * neighbourhood[7] == 0)
								&& (neighbourhood[1] * neighbourhood[3] * neighbourhood[7] == 0)) {
								deleteList.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deleteList.size() == 0)
			break;
		for (size_t i = 0; i < deleteList.size(); i++) {
			Point tem;
			tem = deleteList[i];
			uchar* data = srcImg.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deleteList.clear();

		inOddIterations = !inOddIterations;
	}
}


void cloudvisual(ptrtype cloud, const char* name)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer(name));

	pcl::visualization::PointCloudColorHandlerGenericField<ptype> fildColor(cloud, "z"); // 按照z字段进行渲染

	viewer->addPointCloud<ptype>(cloud, fildColor, "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // 设置点云大小

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
void cloudvisual2(ptrtype src, ptrtype tgt, const char* name)
{
	//创建视窗对象并给标题栏设置一个名称“3D Viewer”并将它设置为boost::shared_ptr智能共享指针，这样可以保证指针在程序中全局使用，而不引起内存错误
	pcl::visualization::PCLVisualizer viewer(name);
	//设置视窗的背景色，可以任意设置RGB的颜色，这里是设置为黑色
	viewer.setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<ptype> target_color(tgt, 0, 255, 0);

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	//将点云添加到视窗对象中，并定一个唯一的字符串作为ID 号，利用此字符串保证在其他成员中也能标志引用该点云，多次调用addPointCloud可以实现多个点云的添加，每调用一次就会创建一个新的ID号，如果想更新一个已经显示的点云，先调用removePointCloud（），并提供需要更新的点云ID 号，也可使用updatePointCloud
	viewer.addPointCloud<ptype>(tgt, target_color, "target cloud",1);
	//用于改变显示点云的尺寸，可以利用该方法控制点云在视窗中的显示方法,1设置显示点云大小
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");

	pcl::visualization::PointCloudColorHandlerCustom<ptype> source_color(src, 255, 0, 0);
	//将点云添加到视窗对象中，并定一个唯一的字符串作为ID 号，利用此字符串保证在其他成员中也能标志引用该点云，多次调用addPointCloud可以实现多个点云的添加，每调用一次就会创建一个新的ID号，如果想更新一个已经显示的点云，先调用removePointCloud（），并提供需要更新的点云ID 号，也可使用updatePointCloud
	viewer.addPointCloud<ptype>(src, source_color, "source cloud",2);
	//用于改变显示点云的尺寸，可以利用该方法控制点云在视窗中的显示方法,1设置显示点云大小
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

pcl::PointCloud<pcl::PointXYZI>::Ptr vegetfilter(const char* your_ALS_file_path)
{

	std::string lasFile_1 = convertFilePath(your_ALS_file_path);
	//打开las文件
	LASreadOpener lasreadopener_1;
	lasreadopener_1.set_file_name(lasFile_1.c_str());
	LASreader* lasreader_A = lasreadopener_1.open();
	size_t count_1 = lasreader_A->header.number_of_point_records;
	pcl::PointCloud<pcl::PointXYZI>::Ptr als_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	long long i_1 = 0;
	while (lasreader_A->read_point()) //&&lasreader_A->point.get_classification()==2)
	{
		int a = lasreader_A->point.get_classification();
		if (a == 2)
		{
			pcl::PointXYZI als_point;
			als_point.x = lasreader_A->point.get_x();
			als_point.y = lasreader_A->point.get_y();
			als_point.z = lasreader_A->point.get_z();
			als_point.intensity = lasreader_A->point.get_intensity();
			//als_cloud->points[i_1].x = lasreader_A->point.get_x();
			//als_cloud->points[i_1].y = lasreader_A->point.get_y();
			//als_cloud->points[i_1].z = lasreader_A->point.get_z();
			//als_cloud->points[i_1].intensity = lasreader_A->point.get_intensity();
			als_cloud->points.push_back(als_point);
			++i_1;
		}
		
	}
	als_cloud->resize(i_1);
	als_cloud->width = i_1;
	als_cloud->height = 1;
	als_cloud->is_dense = false;
	cout << "读取ALS点云数量:" << i_1 << endl;
	return als_cloud;
}

int main()
{

	/**************     机载点云处理         ****************/
	pcl::PointCloud<pcl::PointXYZI>::Ptr als_cloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::PointCloud<pcl::PointXYZI>::Ptr Acloud(new pcl::PointCloud<pcl::PointXYZI>);
	///读取las文件///////
	cout << "设置输入机载文件" << endl;
	string in_als;
	cin >> in_als;
	string als_path = in_als + ".las";
	const char* your_ALS_file_path = als_path.c_str();
	Acloud=vegetfilter(your_ALS_file_path);
	
	
	//半径滤波
	pcl::RadiusOutlierRemoval<ptype> outrem;  //创建滤波器
	outrem.setInputCloud(Acloud);    //设置输入点云
	outrem.setRadiusSearch(1);     //设置半径为0.5的范围内找临近点
	outrem.setMinNeighborsInRadius(10); //设置查询点的邻域点集数小于10的删除
	outrem.setNegative(false);
	// apply filter
	outrem.filter(*als_cloud);     //执行条件滤波   在半径为0.8 在此半径内必须要有两个邻居点，此点才会保存
	//std::cerr << "Cloud after filtering" << endl;
	//std::cerr << mls_cloud->size() << endl;
	cloudvisual(als_cloud, "滤波");
	///提取格网内部最高点云///////
	//提取点云最值
	pcl::PointXYZI min_als;
	pcl::PointXYZI max_als;
	pcl::getMinMax3D(*als_cloud, min_als, max_als);
	//输入格网间隔
	float Agrid_distance;
	cerr << "输入ALS格网间隔值:" << endl;
	cin >> Agrid_distance;

	//计算区域内格网XYZ方向数量
	cerr << "X方向最大值:" << max_als.x << endl;
	cerr << "X方向最小值:" << min_als.x << endl;
	int width_als = int((max_als.x - min_als.x) / Agrid_distance) + 1;

	cerr << "Y方向最大值:" << max_als.y << endl;
	cerr << "Y方向最小值:" << min_als.y << endl;
	int height_als = int((max_als.y - min_als.y) / Agrid_distance) + 1;

	cerr << "区域最大高差:" << max_als.z - min_als.z << endl;

	//构建二维平面格网
	flat_grid **voxel_2 = new flat_grid*[width_als];
	for (int i = 0; i < width_als; ++i)
		voxel_2[i] = new flat_grid[height_als];
	int row_als, col_als;
	for (size_t i = 0; i < als_cloud->points.size(); i++)
	{
		row_als = int((als_cloud->points[i].x - min_als.x) / Agrid_distance);
		col_als = int((als_cloud->points[i].y - min_als.y) / Agrid_distance);
		voxel_2[row_als][col_als].indexID.push_back(i);
		if (voxel_2[row_als][col_als].grayScale < 1)
			voxel_2[row_als][col_als].grayScale++;
	}
	cerr << "格网数量：" << width_als * height_als << endl;
	int count_grid = 0;
	vector<int>pointIndices_als;
	float roofHeight = 0;
	cerr << "输入屋顶高程:" << endl;
	cin >> roofHeight;
	//提取屋顶边沿点云
	for (int i = 0; i < width_als; i++)
	{
		for (int j = 0; j < height_als; j++)
		{
			if (voxel_2[i][j].grayScale == 1)
			{
				count_grid++;
				pcl::PointCloud<pcl::PointXYZI>::Ptr voxelPointCloudPtr(new pcl::PointCloud<pcl::PointXYZI>);   //构建格网点云集
				voxelPointCloudPtr->width = voxel_2[i][j].indexID.size();
				voxelPointCloudPtr->height = 1;
				voxelPointCloudPtr->is_dense = false;
				voxelPointCloudPtr->resize(voxelPointCloudPtr->width * voxelPointCloudPtr->height);
				for (size_t k = 0; k < voxelPointCloudPtr->points.size(); k++)     //读取格网点云数据
				{

					voxelPointCloudPtr->points[k].x = als_cloud->points[voxel_2[i][j].indexID[k]].x;
					voxelPointCloudPtr->points[k].y = als_cloud->points[voxel_2[i][j].indexID[k]].y;
					voxelPointCloudPtr->points[k].z = als_cloud->points[voxel_2[i][j].indexID[k]].z;
				}
				pcl::PointXYZI min;
				pcl::PointXYZI max;
				pcl::getMinMax3D(*voxelPointCloudPtr, min, max);
				voxel_2[i][j].region_difheight = max.z - min_als.z;
				for (size_t k = 0; k < voxelPointCloudPtr->points.size(); k++)     //读取格网点云数据
				{
					if (voxelPointCloudPtr->points[k].z - min_als.z >= roofHeight) pointIndices_als.push_back(voxel_2[i][j].indexID[k]);;
				}
			}
		}
	}
	cout << "输出格网数：" << count_grid << endl;
	pcl::PointCloud<pcl::PointXYZI>::Ptr Acloud_flitered(new pcl::PointCloud<pcl::PointXYZI>);
	boost::shared_ptr<std::vector<int>> index_Aptr = boost::make_shared<std::vector<int>>(pointIndices_als);
	pcl::ExtractIndices<pcl::PointXYZI> Aextract;
	// Extract the inliers
	Aextract.setInputCloud(als_cloud);
	Aextract.setIndices(index_Aptr);
	Aextract.setNegative(false);//如果设为true,可以提取指定index之外的点云
	Aextract.filter(*Acloud_flitered);
	cerr << "输出点云数量:" << Acloud_flitered->size() << endl;
	

	cloudvisual(Acloud_flitered,"屋顶面提取");

	for (int i = 0; i < width_als; ++i)
		delete[] voxel_2[i];
	delete[] voxel_2;
	
	return (0);
}


