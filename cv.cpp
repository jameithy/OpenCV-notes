#include <iostream>
#include <vector>
#include <opencv2/optflow/rlofflow.hop>
#define cv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgu>
#endif




int main()
{
	cv::Mat out;
	cv::Mat img =cv::imread("file path");

	// file path, color, or frame nad color
	cv::Mat img = cv::imread("");

	// python args with new syntax
	cv::imshow("window", img);
	
	//img, out, thresh 1, thresh 2, how specific, gradiant
	cv::Canny(img,out,50, 200,5,1);
	
	//img, out img, block size, k size(Sobel), border type for pixel exploration
	cv::cornerEigenValsAndVecs(img,out,1,1,1);
	
	cv::Mat harringOut;
	cv::outMinEigen;
	cv::outEigenValVec;
	
	
	
	
	//in img, out img, float is how
	cv::cornerHaris(img, harrisOut,3,1,0.0008,1);

	//img, out img, block size
	cv::cornnerMinEigenVal(img,outMinEigen,3,3);
	cv::cornnerEigenValAndVec(img,toutmineigen,3,3);



	for(int i = 0; i<img.cols; i++)
	{
		for(int k = 0; k<img.rows;k++)
		{
			if(harrisOut.at<float>(cv::Point(i,j))>0.0005)
			{
				out.at<cv::Vec3b>(cv::Point(i,j)) == cv::Vec3b(0,255,255);
			}

		}
	}
	
	

	//Good Features To Track
	//input img, output img, quality level, min distance, mask, block size, harris cornners, k
	//
	cv::Mat out;
	cv::goodFeaturesToTrach(img, out, 1.0f,1.0f,mask,3,true,0.04);
	//returns CV_32FC2
	
	std::cout<<corners.size()<<std::endl;
	

	st::count<<out.size()<<std::endl;
	for(int i = 0;i<out.row;i++)
	{
		int xval = out.at<<cv::Vec2f>(cv::Point(i,0))[0];
		int yval = out.at<<cv::Vec2>(cv::Point(1,0))[1];
		cv::circle(out,cv::Point(xval, yval),5,cv::Scalar(0,255,255),cv::FILLED);


	}


	//Open Window
	//name, flags form documentation
	cv::namedWindow(window,cv::WINDOW_NORMAL);
	cv::imshow("window",img);

	cv::moveWindow("window",300,300);


	//create taskbar
	
	int value = 35;
	cv::createTrackbar("OBJECT", "window",&value,100,callback);

	cv::Mat img2;
	while(1)
	{
		img2 = img.clone;
		cv::putText(img2,std::to_sting(value),cv::Point(100,100),2,2,cv::Scalar(0,255,255,4));
		cv::imshow("window",img);
		
		//Track bar pos
		std::cout<<cv::getTrackbarPosition("Value","window")<<std::endl;
			cv::waitkey(0);

	}

	//add all images into one, mask optinal
	cv::accumulate(img, out,mask);

	


	//normilize
	cv::Mat in1 = (cv::Mat_<float>(3,3)<<20,10,5,50,25,2,55,24,12);
	
	cv::normilize(in1, dst, 0,255,cv::NORM_MINMAX);  

	//sort
	// last code
	// only grey scale
	cv::sort(in, out, cv::SORT_EVRY_ROW+cv::SORT_DECENDING);




	//Phase Vector2D rotation angle detection
	cv::Mat _j = (cv::Mat_<float>(2,2)<< 4,8,1,2);
	for(int i =0; i<2; i++);
	{
		for(int k = 0;k<2;k++)
		{
			std::cout<<std::atan2(k<float>(i,k),i<float>(i,k)<<std::endl;
		}
	}

	//out puts array of the angles (MUST SET BOOL AS TRUE FOR NON RADIAN)
	cv::phase(_i, _j, result, true);

	//Mahalanobis	
	
	//Perspective Transform
	std::vector<cv::Point2f> srcPoints;
	srcPoints.push_back(cv::Ponts2f(1,2));

	cv::Mat transformMat = (cv::Mat_<double>(3,3)<<1,2,0,3,0,1,6,0,1);












	//Object Tracking (USE DIS OPTICAL FLOW)
	cv::Ptr<cv::DISOpticalFLow> disOpticalFlow = cv::DISOpticalFlow::create();
	cv::Ptr<cv::DenseROLFOpticalFLow> disOpticalFlow2 = cv::DISOpticalFlow::create();
	cv::Ptr<cv::VariationalRefinement> varRefine = cv::VariatinolRefinement::create();
	cv::Mat frame, prevFrame, flow;
	cv::nameOfWindow("Output", 0);
	cv::VideoCapture cap ("file path");

	disOpticalFlow->setFlowScale(100);
	disOpticalFlow->setPatchStride(8);
	//for noisy background
	disOpticalFlow->getUseMeanNormalization(0);

	disOpticalFlow->getUseSpatialPropogation(0);
	//faster objects
	disOpticalFlow->setUseSpatialPropagation(0);
	//shorter for longer lines
	disOpticalFlow->getVariationalRefinementAlpha(1);
	//longer for shorter lines
	disOpticalFlow->getVariationalRefinementGamma(1);



	//smoothness
	varRefine->getAlpha();
	//color constancy (not for gray scale)
	varRefine->getDelta(1);
	//higher number for accuracy but less speed
	varRefine->setFixedPointIterations(100);
	//gradiant (better is closer to 2) (foat)
	varRefine->getGamma();








	cv::Point2f updatePositionAlongRectangle(cv::Point2f currentPosition, cv::Point2f& direction, const cv::Rect& rect);
    currentPosition += direction;

    if (currentPosition.x <= rect.x || currentPosition.x >= rect.x + rect.width) {
        direction.x = -direction.x;
    }
    if (currentPosition.y <= rect.y || currentPosition.y >= rect.y + rect.height) {
        direction.y = -direction.y;
    }

    return currentPosition;
}

int main()
	{

    cv::namedWindow("Kalman Filter Tracking",0);
    cv::KalmanFilter KF(4, 2, 0);

    KF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                           0, 1, 0, 1,
                           0, 0, 1, 0,
                           0, 0, 0, 1);

    KF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0,
													 0, 1, 0, 0);
 


    setIdentity(KF.processNoiseCov, cv::Scalar(1e-4));

    setIdentity(KF.measurementNoiseCov, cv::Scalar(1e-1));

    setIdentity(KF.errorCovPost, cv::Scalar(1));

    KF.statePost = (cv::Mat_<float>(4, 1) << 0, 0, 0, 0);

    int width = 640;
    int height = 480;
    cv::Mat frame(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Rect rectangle(100, 100, 440, 280);

    cv::Point2f truePosition(rectangle.x + rectangle.width / 2, rectangle.y + rectangle.height / 2);
    cv::Point2f direction(5, 5);

    cv::Point2f predictedPosition;

    while (true) {
        frame.setTo(cv::Scalar(0, 0, 0));

        truePosition = updatePositionAlongRectangle(truePosition, direction, rectangle);

        cv::Mat prediction = KF.predict();
        predictedPosition = cv::Point2f(prediction.at<float>(0), prediction.at<float>(1));

        cv::Mat measurement = (cv::Mat_<float>(2, 1) << truePosition.x, truePosition.y);
        KF.correct(measurement);

        cv::Mat future_state = KF.statePost.clone();
        for (int i = 0; i < 10; ++i) { // predicting 10 steps ahead
            future_state = KF.transitionMatrix * future_state;
        }
        cv::Point futurePredictPt(future_state.at<float>(0), future_state.at<float>(1));

        cv::rectangle(frame, rectangle, cv::Scalar(255, 0, 0), 2); // Rectangle in blue
        cv::circle(frame, truePosition, 10, cv::Scalar(255, 255, 255), -1); // True position in white
        cv::circle(frame, predictedPosition, 10, cv::Scalar(0, 255, 0), -1); // Predicted position in green
        cv::circle(frame, futurePredictPt, 10, cv::Scalar(0, 0, 255), -1); // Predicted position in green

        cv::imshow("Kalman Filter Tracking", frame);

        cv::waitKey(0);
    }

    return 0;
}
	





	while(1)
	{
		cap >> frame;
		if(!prevFrame.empty && cnt%14==0)
		{
			//gray scale images for calc()
			//cv::ColorTwoPlane(frame,frame,cv::COLOR_BGR2GRAY);
			isObticalFlow->calc(prevFrame, frame, flow);
			//varOpticalFlow->calc(prevFrame,frame,flow);
			for (int y=0;y<frame.rows;y+=50) {

				for (int x=0;x<frame.cols;x+=50) {

					cv::Point2f floatAtPoints = flow.at<cv::Point2d>(y,x);
					cv::line(frame, cv::Points(x,y), cv::Points(cvRound(x + flowAtPoint.x), cvRound(y+flowAtPoint.y)),cv::Scalar_());
					cv::circle(frame,cv::Points(x,y),1,cv::Scalar(0,255,0), 1);

				}

			}
		}


		prevFrame = frame.clone();


		cv::imshow("Output", frame);
		cv::waitkey(1);		
	}
	






	cv::waitkey(1);
	return 0;
}

void callback(int a, void*)
{
	std::count<<a<<std::endl;
}
