#include <QCoreApplication>
#include <QDir>
#include <QTime>
#include <QtCore>

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/ml.hpp>

#include <stdio.h>
#include <stdlib.h>
#include<iostream>

//using namespace std;

#define characterMinWidthThreshold 9
#define characterMaxWidthThreshold 50
#define sizeOfTrainImage 20
#define trainPositiveLabel 1
#define trainNegativeLabel 2

bool IsGrayImage( cv::Mat img ) // returns true if the given 3 channel image is B = G = R
{
    cv::Mat dst;
    cv::Mat bgr[3];
    split( img, bgr );
    absdiff( bgr[0], bgr[1], dst );

    if(countNonZero( dst ))
        return false;

    absdiff( bgr[0], bgr[2], dst );
    return !countNonZero( dst );
}

void Preprocess(cv::Mat &image, cv::Mat &invertedGrayImage){
//    convert to gray, resize, invert, detect if image is colored so plate detection is needed, padding

    cv::Mat grayImage;
    if(image.channels() == 1)
        grayImage = image;
    else if (IsGrayImage(image)) // convert 3 channels image to 1 channel
        cv::cvtColor(image, grayImage, CV_BGR2GRAY);
    else
    {
//        some of input images are colored and need another process to find the plate image
//        detectThePlate();
        cv::cvtColor(image, grayImage, CV_BGR2GRAY);
    }

    subtract(cv::Scalar::all(255),grayImage,invertedGrayImage);
    resize(invertedGrayImage, invertedGrayImage, cv::Size(165, 30));
    copyMakeBorder( invertedGrayImage, invertedGrayImage, 10, 10, 10, 10, cv::BORDER_CONSTANT, cv::Scalar( 0,0,0) );

}

void FindContoursOfPlate(cv::Mat &invertedGrayImage, std::vector<cv::Rect> &selectedRect, std::vector<cv::Rect> &corruptedcharacterRect){

    cv::Mat imageCanny;
    cv::Canny(invertedGrayImage, imageCanny, 100, 200, 3);
    std::vector<std::vector<cv::Point>> contoursBS;

    std::vector<cv::Vec4i> hierarchyBS;


    double minPixel, maxPixel;
    cv::minMaxLoc(invertedGrayImage, &minPixel, &maxPixel);



    findContours(imageCanny, contoursBS, hierarchyBS, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


    cv::Rect contourRect;

//    loop on the countours from gray inverted image to classify each of them with their width
    for (int cIndex = 0; cIndex < contoursBS.size(); cIndex++)
    {
        cv::Scalar color_bs = cv::Scalar(0, 255, 0);
        contourRect = cv::boundingRect(contoursBS[cIndex]);


        if ( contourRect.width >= characterMinWidthThreshold && contourRect.width <= 25) // there should be a healthy or corrupted character !
        {

            selectedRect.push_back(contourRect);

        }
        else if ( contourRect.width > 25 && contourRect.height >= 7) // there can be more than one character
        {
            corruptedcharacterRect.push_back(contourRect);
            std::vector<cv::Vec4i> hierarchyBsAgain;
            cv::Mat contourRectImage;
            invertedGrayImage(contourRect).copyTo(contourRectImage);
//            threshold gray image to separate characters


            threshold(contourRectImage, contourRectImage, maxPixel - 35, 255, 0);
            cv::Canny(contourRectImage, contourRectImage, 100, 200, 3);
            std::vector<std::vector<cv::Point>> contoursAgain;

//            find contours on thresholded big contour again
            findContours(contourRectImage, contoursAgain, hierarchyBsAgain, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
            bool contourRectSave[contourRect.width] = { false };

//            loop on each new contour to classify them with their width
            for ( int cIndexAgain = 0; cIndexAgain < contoursAgain.size(); cIndexAgain++ )
            {
                cv::Rect contourRectAgain = cv::boundingRect(contoursAgain[cIndexAgain]);

//                that really should be one character or one corrupted one
                if ( contourRectAgain.width >= characterMinWidthThreshold && contourRectAgain.height >= 7 && contourRect.width < characterMaxWidthThreshold)
                {
                    cv::Rect tmpRect;
                    tmpRect.x = contourRectAgain.x + contourRect.x;
                    tmpRect.y = contourRect.y;
                    tmpRect.width = contourRectAgain.width;
                    tmpRect.height = contourRect.height;

                    bool selectedOrNot = false;
                    for (int rectWIndex = 0; rectWIndex < contourRectAgain.width ; rectWIndex++)
                    {
                        if ( contourRectSave[contourRectAgain.x+rectWIndex] != true )
                        {
                            contourRectSave[contourRectAgain.x+rectWIndex] = true;
                            selectedOrNot = true;
                        }
                    }
                    if (selectedOrNot)
                        selectedRect.push_back(tmpRect);



                }
                else if ( contourRectAgain.width >= characterMaxWidthThreshold && contourRectAgain.height >= 7)
                {
                    cv::Rect tmpRect;
                    tmpRect.x = contourRectAgain.x + contourRect.x;
                    tmpRect.y = contourRect.y;
                    tmpRect.width = contourRectAgain.width;
                    tmpRect.height = contourRect.height;

                    bool corruptedOrNot = false;
                    for (int rectWIndex = 0; rectWIndex < contourRectAgain.width ; rectWIndex++)
                    {
                        if ( contourRectSave[contourRectAgain.x+rectWIndex] != true )
                        {
                            contourRectSave[contourRectAgain.x+rectWIndex] = true;
                            corruptedOrNot = true;

                        }
                    }
                    if (corruptedOrNot)
                        corruptedcharacterRect.push_back(tmpRect);

                }
            }

//            int countFalses = 0;
//            save regions destroyed in thresholded image
//            for ( int contourRectSaveIndex = 0 ; contourRectSaveIndex < contourRect.width; contourRectSaveIndex++)
//            {
//                if ( contourRectSave[contourRectSaveIndex] == false && countFalses < characterMinWidthThreshold )
//                    countFalses++;
//                if ( contourRectSave[contourRectSaveIndex] == true && countFalses >= characterMinWidthThreshold )
//                {
//                    // here we have a region bigger than a character which is not detected as a countour in thresholded image
//                    Rect tmpRect;
//                    tmpRect.x = contourRect.x + contourRectSaveIndex - countFalses;
//                    tmpRect.width = countFalses;
//                    tmpRect.height = contourRect.height;
//                    tmpRect.y = contourRect.height;
//                    if ( countFalses > characterMaxWidthThreshold )
//                        corruptedcharacterRect.push_back(tmpRect);
//                    else
//                        selectedRect.push_back(tmpRect);

//                }
//                if ( contourRectSave[contourRectSaveIndex] == true && countFalses < characterMinWidthThreshold )
//                {
//                    countFalses = 0;
//                }
//                if ( contourRectSave[contourRectSaveIndex] == false && countFalses >= characterMinWidthThreshold)
//                {
//                    countFalses++;
//                }
//                if ( contourRectSaveIndex == contourRect.width-1 && countFalses >= characterMinWidthThreshold )
//                {
//                    Rect tmpRect;
//                    tmpRect.x = contourRect.x + contourRectSaveIndex - countFalses;
//                    tmpRect.width = countFalses;
//                    tmpRect.height = contourRect.height;
//                    tmpRect.y = contourRect.height;
//                    if ( countFalses > characterMaxWidthThreshold )
//                        corruptedcharacterRect.push_back(tmpRect);
//                    else
//                        selectedRect.push_back(tmpRect);
//                }

//            }
        }

    }
}

void ModifyRectRegionSize(cv::Mat &image){
    if ( image.cols >= image.rows )
    {
        int pad = (image.cols - image.rows)/2;
        copyMakeBorder( image, image, pad, image.cols-image.rows-pad, 0, 0, cv::BORDER_CONSTANT, cv::Scalar( 0,0,0) );
        resize( image, image, cv::Size(sizeOfTrainImage,sizeOfTrainImage));
    }
    else if ( image.cols < image.rows )
    {
        int pad = (image.rows - image.cols)/2;
        cv::copyMakeBorder( image, image, 0, 0, pad, image.rows-image.cols-pad, cv::BORDER_CONSTANT, cv::Scalar( 0,0,0) );
        resize( image, image, cv::Size(sizeOfTrainImage,sizeOfTrainImage));

    }

}


void ConvertVectortoMatrix(std::vector<std::vector<float> > &descriptor, cv::Mat &descriptorMat)
{

    int descriptor_size = descriptor[0].size();

    for(int i = 0;i<descriptor.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            descriptorMat.at<float>(i,j) = descriptor[i][j];
        }
    }

}

void GetSVMParams(cv::ml::SVM *svm)
{
    std::cout << "Kernel type     : " << svm->getKernelType() << std::endl;
    std::cout << "Type            : " << svm->getType() << std::endl;
    std::cout << "C               : " << svm->getC() << std::endl;
    std::cout << "Degree          : " << svm->getDegree() << std::endl;
    std::cout << "Nu              : " << svm->getNu() << std::endl;
    std::cout << "Gamma           : " << svm->getGamma() << std::endl;
}

void SVMtrainAndTest(cv::Mat &trainMat,std::vector<int> &trainLabels, cv::Mat &testResponse,cv::Mat &testMat){

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setGamma(0.50625);
    svm->setC(12.5);
    svm->setKernel(cv::ml::SVM::RBF);
    svm->setType(cv::ml::SVM::C_SVC);
    cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(trainMat, cv::ml::ROW_SAMPLE, trainLabels);
    svm->train(td);
    svm->save("model4.yml");
    svm->predict(testMat, testResponse);
    GetSVMParams(svm);

}

void SVMTest(cv::Mat &testResponse,cv::Mat &testMat)
{
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm = cv::Algorithm::load<cv::ml::SVM>("model4.yml");
    svm->predict(testMat, testResponse);
    GetSVMParams(svm);

}

//    here we write a results directory, which plates and their results are stored
void WriteTheResults(std::vector<cv::String> &testFileNames, std::vector<cv::Rect> &testSetSelectedRect, std::vector<cv::Rect> &testSetCorruptedRect, cv::Mat &testResponse, std::vector<int> &testIndexSelected, std::vector<int> &testIndexCorrupted)
{

    for(size_t filenameIndex = 0; filenameIndex < testFileNames.size(); ++filenameIndex)
    {

        cv::Mat testImage = imread(testFileNames[filenameIndex],CV_LOAD_IMAGE_GRAYSCALE);
        cv::resize(testImage, testImage, cv::Size(165, 30));

        cv::cvtColor(testImage, testImage,  cv::COLOR_GRAY2BGR);

        if(!testImage.data)
            std::cerr << "Problem loading image!!!" << endl;
        std::vector <cv::Rect> thisFramePositiveRects, thisFrameCorruptedRects;
        std::vector <int> thisFrameResults;

        for ( int rectCIndex = 0; rectCIndex < testIndexCorrupted.size() ; rectCIndex++ )
        {

            if ( testIndexCorrupted[rectCIndex] == filenameIndex )
            {
                cv::Rect resRect = testSetCorruptedRect[rectCIndex];
                resRect.y = 0;
                resRect.height=30;
                resRect.x = resRect.x - 10;
                thisFrameCorruptedRects.push_back(resRect);


            }
            else
                continue;

        }

        //
        cv::Rect x,y;
        cv::Rect z = x & y;
//        std::vector<bool> thisFrameCWrite, thisFramePWrite;
        for ( int rectSIndex = 0; rectSIndex < thisFrameCorruptedRects.size() ; rectSIndex++ )
        {
            for ( int rectSSIndex = rectSIndex ; rectSSIndex < thisFrameCorruptedRects.size() ; rectSSIndex++ )
            {
                cv::Rect r1 = thisFrameCorruptedRects[rectSIndex];
                cv::Rect r2 = thisFrameCorruptedRects[rectSSIndex];
                cv::Rect r3 = r1 & r2;
                cv::Rect r4 = r1 | r2;
                if ( r3.area() > 0 )
                    rectangle(testImage, r4 , cv::Scalar(0, 0, 255), 1);
//                if ( r3.area() == r1.area())
//                {
//                    rectangle(testImage, r2, Scalar(0, 0, 255), 1);
//                }
//                if ( r3.area() == r2.area())
//                {
//                    rectangle(testImage, r1, Scalar(0, 0, 255), 1);

//                }

            }

        }

        for ( int rectSIndex = 0; rectSIndex < testIndexSelected.size() ; rectSIndex++ )
        {
            if ( testIndexSelected[rectSIndex] == static_cast<int>(filenameIndex) )
            {
                cv::Rect resRect = testSetSelectedRect[rectSIndex];
                resRect.y = 0;
                resRect.height=30;
                resRect.x = resRect.x - 10;

                if(testResponse.at<float>(rectSIndex,0) == 1)
                {   thisFramePositiveRects.push_back(resRect);
                    cv::rectangle(testImage, resRect, cv::Scalar(0, 255, 0), 1);}
                else{
                    thisFrameCorruptedRects.push_back(resRect);
                    cv::rectangle(testImage, resRect, cv::Scalar(0, 0, 255), 1);
                }
//                thisFrameSelectedRects.push_back(resRect);
//                thisFrameResults.push_back(testResponse.at<float>(rectSIndex,0));
            }
            else
                continue;
        }
        // make a result directory

        if ( !QDir("result").exists())
            QDir().mkdir("result");

        std::string fileWriteName = "./result/result_" + std::to_string(filenameIndex) + ".jpg";
        imwrite(fileWriteName, testImage);
    }






}

int main(int argc, char *argv[])
{

    // Arguments: run the code with testset path to test it with pretrained model
    // Or run with 2 pathes to train and test the code. first one should be trainset path
    QCoreApplication a(argc, argv);
    QTime myTimer;
    myTimer.start();
    std::string trainSetPath, testSetPath;
    bool wannaTrain = false;
    if ( argc == 3 )
    {
        trainSetPath = argv[1];
        testSetPath = argv[2];
        wannaTrain = true;
    } else if ( argc == 2 )
    {
        testSetPath = argv[1];
        wannaTrain = false;
    }
    

    cv::HOGDescriptor hog(
                cv::Size(20,20), //winSize
                cv::Size(8,8), //blocksize
                cv::Size(4,4), //blockStride,
                cv::Size(8,8), //cellSize,
                9, //nbins,
                1, //derivAper,
                -1, //winSigma,
                0, //histogramNormType,
                0.2, //L2HysThresh,
                0,//gammal correction,
                64,//nlevels=64
                1);

    // load data
    
    std::vector<cv::Mat> trainSet;
    std::vector<cv::Mat> testSet;
    std::vector<int> trainLabels;
    std::vector<int> testLabels;
    
    if (wannaTrain)
    {
        std::vector<cv::String> testFileNames, trainPositiveFileNames, trainNegativeFileNames;
        QDir dir(QString::fromStdString(trainSetPath));
        dir.setFilter(QDir::Dirs);
        QFileInfoList trainLabelsPath = dir.entryInfoList();
        std::string trainSetPositivePath = trainLabelsPath.at(2).filePath().toStdString();
        std::string trainSetNegativePath = trainLabelsPath.at(3).filePath().toStdString();
        cv::glob(trainSetPositivePath, trainPositiveFileNames);
        cv::glob(trainSetNegativePath, trainNegativeFileNames);
        cv::glob(testSetPath, testFileNames);
        std::vector<std::vector<float> > trainSetDescriptors;
        for(size_t filenameIndex = 0; filenameIndex < trainPositiveFileNames.size(); ++filenameIndex)
        {
            cv::Mat trainImage = imread(trainPositiveFileNames[filenameIndex],CV_LOAD_IMAGE_ANYDEPTH);

            ModifyRectRegionSize(trainImage);
            std::vector<float> descriptors;
            hog.compute(trainImage,descriptors);
            trainSetDescriptors.push_back(descriptors);
            trainLabels.push_back(trainPositiveLabel);

        }

        for(size_t filenameIndex = 0; filenameIndex < trainNegativeFileNames.size(); ++filenameIndex)
        {
            cv::Mat trainImage = imread(trainNegativeFileNames[filenameIndex],CV_LOAD_IMAGE_ANYDEPTH);
            ModifyRectRegionSize(trainImage);
            std::vector<float> descriptors;
            hog.compute(trainImage,descriptors);
            trainSetDescriptors.push_back(descriptors);
            trainLabels.push_back(trainNegativeLabel);

        }


        //test images process:
        std::vector<cv::Rect> testSetSelectedRect, testSetCorruptedRect; // these vectors keep rectangles of characters
        std::vector<int> testIndexSelected, testIndexCorrupted; // these vectors keep which rect is for which test image
        std::vector<std::vector<float> > testSetDescriptors;

        for(size_t filenameIndex = 0; filenameIndex < testFileNames.size(); ++filenameIndex)
        {
            cv::Mat testImage = imread(testFileNames[filenameIndex],CV_LOAD_IMAGE_ANYDEPTH);
            if(!testImage.data)
                std::cerr << "Problem loading image!!!" << endl;

//            check if im  age is Grayscale or RGB
            bool isColored = false;
            cv::Mat testpreprocessedImage;


            Preprocess(testImage, testpreprocessedImage); // preproces step
            int numOfSelected, numOfCorrupted;
            numOfSelected = testSetSelectedRect.size();
            numOfCorrupted = testSetCorruptedRect.size();

            FindContoursOfPlate(testpreprocessedImage, testSetSelectedRect, testSetCorruptedRect);


            std::vector<cv::Rect> thisPlateSelectedRects;
            std::vector<cv::Rect> thisPlateCorruptedRects;
            for (int w = numOfSelected; w < testSetSelectedRect.size() ; w++)
            {
                thisPlateSelectedRects.push_back(testSetSelectedRect[w]);
                testIndexSelected.push_back(filenameIndex);
            }

            for (int w = numOfCorrupted; w < testSetCorruptedRect.size(); w++)
            {
                thisPlateCorruptedRects.push_back(testSetCorruptedRect[w]);
                testIndexCorrupted.push_back(filenameIndex);
            }


            for ( int testRIndex = 0 ; testRIndex < thisPlateSelectedRects.size(); testRIndex++ )
            {
                cv::Mat testSelectedRectImage;
                cv::Rect crop;
                crop.x = thisPlateSelectedRects[testRIndex].x;
                crop.width = thisPlateSelectedRects[testRIndex].width;
                crop.y = 10;
                crop.height = 30;
                testpreprocessedImage(crop).copyTo(testSelectedRectImage);
                ModifyRectRegionSize(testSelectedRectImage);
//                string fileNamew = "./testImg_" + to_string(filenameIndex) + "_" + to_string(testRIndex) + ".jpg";
//                imwrite(fileNamew,testSelectedRectImage);
                std::vector<float> descriptors;
                hog.compute(testSelectedRectImage,descriptors);
                testSetDescriptors.push_back(descriptors);


            }
            for ( int testRIndex = 0 ; testRIndex < thisPlateCorruptedRects.size(); testRIndex++ )
            {
                cv::Mat testCorruptedRectImage;
                cv::Rect crop;
                crop.x = thisPlateCorruptedRects[testRIndex].x;
                crop.width = thisPlateCorruptedRects[testRIndex].width;
                crop.y = 10;
                crop.height = 30;
                testpreprocessedImage(crop).copyTo(testCorruptedRectImage);
                ModifyRectRegionSize(testCorruptedRectImage);
                std::string fileNamew = "./CorruptedImg_" + std::to_string(filenameIndex) + "_" + std::to_string(testRIndex) + ".jpg";
//                imwrite(fileNamew,testCorruptedRectImage);

            }

        }

        int descriptor_size = trainSetDescriptors[0].size();
//        std::cout << "Descriptor Size : " << descriptor_size << endl;

        cv::Mat trainDescriptorMat(trainSetDescriptors.size(),descriptor_size,CV_32FC1);
        cv::Mat testDescriptorMat(testSetDescriptors.size(),descriptor_size,CV_32FC1);

        ConvertVectortoMatrix(trainSetDescriptors,trainDescriptorMat);
        ConvertVectortoMatrix(testSetDescriptors,testDescriptorMat);

        cv::Mat testResponse;
        SVMtrainAndTest(trainDescriptorMat,trainLabels,testResponse,testDescriptorMat);

        WriteTheResults(testFileNames, testSetSelectedRect, testSetCorruptedRect, testResponse, testIndexSelected, testIndexCorrupted);
        std::vector<int> testLabels;

    }

    else
    {
        std::vector<cv::String> testFileNames;


        cv::glob(testSetPath, testFileNames);

//        test images process:
        std::vector<cv::Rect> testSetSelectedRect, testSetCorruptedRect; // these vectors keep rectangles of characters
        std::vector<int> testIndexSelected, testIndexCorrupted; // these vectors keep which rect is for which test image
        std::vector<std::vector<float> > testSetDescriptors;

        for(size_t filenameIndex = 0; filenameIndex < testFileNames.size(); ++filenameIndex)
        {
            cv::Mat testImage = imread(testFileNames[filenameIndex],CV_LOAD_IMAGE_ANYDEPTH);
            if(!testImage.data)
                std::cerr << "Problem loading image!!!" << endl;

//            check if im  age is Grayscale or RGB
            bool isColored = false;
            cv::Mat testpreprocessedImage;


            Preprocess(testImage, testpreprocessedImage); // preproces step
            int numOfSelected, numOfCorrupted;
            numOfSelected = testSetSelectedRect.size();
            numOfCorrupted = testSetCorruptedRect.size();

            FindContoursOfPlate(testpreprocessedImage, testSetSelectedRect, testSetCorruptedRect);


            std::vector <cv::Rect> thisPlateSelectedRects;
            std::vector <cv::Rect> thisPlateCorruptedRects;
            for (int w = numOfSelected; w < testSetSelectedRect.size() ; w++)
            {
                thisPlateSelectedRects.push_back(testSetSelectedRect[w]);
                testIndexSelected.push_back(filenameIndex);
            }

            for (int w = numOfCorrupted; w < testSetCorruptedRect.size(); w++)
            {
                thisPlateCorruptedRects.push_back(testSetCorruptedRect[w]);
                testIndexCorrupted.push_back(filenameIndex);
            }


            for ( int testRIndex = 0 ; testRIndex < thisPlateSelectedRects.size(); testRIndex++ )
            {
                cv::Mat testSelectedRectImage;
                cv::Rect crop;
                crop.x = thisPlateSelectedRects[testRIndex].x;
                crop.width = thisPlateSelectedRects[testRIndex].width;
                crop.y = 10;
                crop.height = 30;
                testpreprocessedImage(crop).copyTo(testSelectedRectImage);
                ModifyRectRegionSize(testSelectedRectImage);
//                string fileNamew = "./testImg_" + to_string(filenameIndex) + "_" + to_string(testRIndex) + ".jpg";
//                imwrite(fileNamew,testSelectedRectImage);
                std::vector<float> descriptors;
                hog.compute(testSelectedRectImage,descriptors);
                testSetDescriptors.push_back(descriptors);


            }
            for ( int testRIndex = 0 ; testRIndex < thisPlateCorruptedRects.size(); testRIndex++ )
            {
                cv::Mat testCorruptedRectImage;
                cv::Rect crop;
                crop.x = thisPlateCorruptedRects[testRIndex].x;
                crop.width = thisPlateCorruptedRects[testRIndex].width;
                crop.y = 10;
                crop.height = 30;
                testpreprocessedImage(crop).copyTo(testCorruptedRectImage);
                ModifyRectRegionSize(testCorruptedRectImage);
//                string fileNamew = "./CorruptedImg_" + to_string(filenameIndex) + "_" + to_string(testRIndex) + ".jpg";
//                imwrite(fileNamew,testCorruptedRectImage);

            }

        }

        int descriptor_size = testSetDescriptors[0].size();
//        cout << "Descriptor Size : " << descriptor_size << endl;


        cv::Mat testDescriptorMat(testSetDescriptors.size(),descriptor_size,CV_32FC1);


        ConvertVectortoMatrix(testSetDescriptors,testDescriptorMat);
        cv::Mat testResponse;

        SVMTest(testResponse, testDescriptorMat);

        WriteTheResults(testFileNames, testSetSelectedRect, testSetCorruptedRect, testResponse, testIndexSelected, testIndexCorrupted);

    }

    int nMilliseconds = myTimer.elapsed();
    qDebug() << "process time:" << nMilliseconds;
    return 0;
}
