#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#define img_num 300
using namespace cv;
using namespace std;

//把图像归一化为0-255
Mat norm_0_255(const Mat& src){
  Mat dst;
  switch(src.channels()){
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
  }
  return dst;
}

//将给定的图像转化为行矩阵
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0){
  // 样本数量
  size_t n = src.size();
  //没有样本，返回空矩阵
  if (n == 0)
    return Mat();

  //样本的维数
  size_t d = src[0].total();

  Mat data(n, d, rtype);

  //拷贝数据
  for (int i = 0; i < n; i++){
    if(src[i].empty()){
      string error_message = format("Image number %d was empty, please check your input data.", i);
      CV_Error(CV_StsBadArg, error_message);
    }

    //确保数据能被reshape
    if(src[i].total() != d){
      string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
      CV_Error(CV_StsBadArg, error_message);
    }
    Mat xi = data.row(i);
    //转化为1行，n列的格式
    if(src[i].isContinuous()){
      src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
    }
    else {
      src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
    }
  }
  return data;
}

int main(int argc, const char* argv[]){
  vector<Mat> db;
  for(int i=1; i<11;i++){

    string filename = format("Img/test/%d.pgm", i);
    db.push_back(imread(filename, IMREAD_GRAYSCALE));
  }
  // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);

    // PCA算法保持5主成分分量
    int num_components = 5;

    //执行pca算法
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    //copy  pca算法结果
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();
        //均值脸
    imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));

    //五个特征脸
    imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));
    imshow("pc4", norm_0_255(pca.eigenvectors.row(3)).reshape(1, db[0].rows));
    imshow("pc5", norm_0_255(pca.eigenvectors.row(4)).reshape(1, db[0].rows));

    while(1)
        waitKey(0);

    // Success!
    return 0;

}
