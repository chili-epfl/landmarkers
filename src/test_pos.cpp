// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    

    This example is essentially just a version of the face_landmark_detection_ex.cpp
    example modified to use OpenCV's VideoCapture object to read from a camera instead 
    of files.


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <iostream>
#include <fstream>

using namespace dlib;
using namespace std;

int main(int argc, char** argv)
{

    if (argc == 1){
        cout << "need destination to record the activity" << endl;
        return 0;
    }

    try
    {

        ofstream myfile;
        auto filename = argv[1];
        myfile.open (filename);
        std::vector<full_object_detection> contacts;

        cv::VideoCapture cap(0);
        image_window win;

        // Load face detection and pose estimation models.
        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        std::vector<float> EMA(6,1);
        float threshold = 1;
        float mu = 0.1;
        float eps = 0.00000001;
        float t = 1;

        // Grab and process frames until the main window is closed by the user.
        while(!win.is_closed())
        {
            // Grab a frame
            cv::Mat temp;
            cap >> temp;
            // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
            // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
            // long as temp is valid.  Also don't do anything to temp that would cause it
            // to reallocate the memory which stores the image as that will make cimg
            // contain dangling pointers.  This basically means you shouldn't modify temp
            // while using cimg.
            cv_image<bgr_pixel> cimg(temp);

            // Detect faces 
            std::vector<rectangle> faces = detector(cimg);
            // Find the pose of each face.
            std::vector<full_object_detection> shapes;

            std::vector<rectangle> noses;
            std::vector<rectangle> sides;

            for (unsigned long i = 0; i < faces.size(); ++i){
                shapes.push_back(pose_model(cimg, faces[i]));

                //get noses
                auto x = pose_model(cimg, faces[i]).part(30).x();
                auto y = pose_model(cimg, faces[i]).part(30).y();
                noses.push_back(centered_rect(point(x-2,y-2),8,8));

                //get lefts
                auto x1 = pose_model(cimg, faces[i]).part(2).x();
                auto y1 = pose_model(cimg, faces[i]).part(2).y();
                sides.push_back(centered_rect(point(x1-2,y1-2),8,8));

                //get lefts
                auto x2 = pose_model(cimg, faces[i]).part(14).x();
                auto y2 = pose_model(cimg, faces[i]).part(14).y();

                //get down
                auto x3 = pose_model(cimg, faces[i]).part(2).x();
                auto y3 = pose_model(cimg, faces[i]).part(2).y();

                //get up
                auto x4 = 0.5*( pose_model(cimg, faces[i]).part(21).x() + pose_model(cimg, faces[i]).part(22).x() );
                auto y4 = 0.5*( pose_model(cimg, faces[i]).part(21).y() + pose_model(cimg, faces[i]).part(22).y() );


               sides.push_back(centered_rect(point(x2-2,y2-2),8,8));

                float a = sqrt((x1-x)*(x1-x) + (y1-y)*(y1-y));
                float b = sqrt((x2-x)*(x2-x) + (y2-y)*(y2-y));
                float c = sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2));

                auto estwest = (a-b)/c; // -1<score<1

                auto l = a*c/(a+b);
                auto xl = x1+(x2-x1)*l/c;
                auto yl = y1+(y2-y1)*l/c;

                auto sh = (yl-y)/abs(y-yl);
                auto h = sqrt((y-yl)*(y-yl) + (x-xl)*(x-xl));
                auto southnorth = sh*h/a;


                auto look_left = estwest>0.2;
                auto look_right = estwest<-0.2;
                auto look_up = southnorth>0.2;
                auto look_down = southnorth<-0.2;

                auto contact = true;

                if(look_left){
                    //cout << "face " << i << " look left" << endl; 
                    contact=false;
                }
                if(look_right){
                    //cout << "face " << i << " look right" << endl; 
                    contact=false;
                }
                if(look_up){
                    //cout << "face " << i << " look up" << endl; 
                    contact=false;
                }
                if(look_down){
                   // cout << "face " << i << " look down" << endl; 
                    contact=false;
                }

                if(contact){
                    //cout << "contact !!" << endl;
                    contacts.push_back(pose_model(cimg, faces[i]));
                }

                // Compute size of the head
                float d = sqrt((x3-x4)*(x3-x4) + (y3-y4)*(y3-y4));
                auto size = c*d;
                cout << int(size/1000) << endl;


                // Compute novelty

                std::vector<float> X(6,0);
                X[0] = float(look_right);
                X[1] = float(look_left);
                X[2] = float(look_up);
                X[3] = float(look_down);
                X[4] = float(contact);
                X[5] = float(faces.size());

                auto temp = EMA;

                for (unsigned int j = 0; j < EMA.size(); ++j){
                    EMA[j] = mu*X[j] + (1-mu)*EMA[j];
                }

                float dist = 0.0;

                for (unsigned int j = 0; j < EMA.size(); ++j){
                    dist += 2*(EMA[j]-temp[j])*(EMA[j]-temp[j]) / ( eps+(EMA[j]+temp[j])*(EMA[j]+temp[j]) );
                }
                if( dist>threshold){
                    //cout <<"novelty ! :"<< dist*sqrt(t) << endl;
                    t = 1;
                }
                else{
                    t++;
                }

            }
            if( faces.size() == 0){

                std::vector<float> X(6,0);
                auto temp = EMA;

                for (unsigned int j = 0; j < EMA.size(); ++j){
                    EMA[j] = mu*X[j] + (1-mu)*EMA[j];
                }

                float dist = 0.0;
                for (unsigned int j = 0; j < EMA.size(); ++j){
                    dist += 2*(EMA[j]-temp[j])*(EMA[j]-temp[j]) / ( eps+(EMA[j]+temp[j])*(EMA[j]+temp[j]) );
                }
                if( dist>threshold){
                    //cout <<"novelty ! :"<< dist*sqrt(t) << endl;
                    t = 1;
                }
                else{
                    t++;
                }
            }


            // Display it all on the screen
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
            for( auto nose : noses)
                win.add_overlay(nose);
            for( auto side : sides)
                win.add_overlay(side);
        }

        cout << "please wait, recording the landmark positions during eye_contacts. it could take a while"<< endl;
        //cout << contacts.size() << endl;

        //auto i = 0;
        for( auto contact : contacts){
            //i++;
            //cout<<i<< "of "<< contacts.size()<<endl;
            for (unsigned long j = 0; j < contact.num_parts(); ++j){
                myfile << contact.part(j).x() << " ";
            }
            for (unsigned long j = 0; j < contact.num_parts(); ++j){
                myfile << contact.part(j).y() << " ";
            }
            myfile << endl;
        }


        myfile.close();
    }
    catch(serialization_error& e)
    {
        cout << "You need dlib's default face landmarking model file to run this example." << endl;
        cout << "You can get it from the following URL: " << endl;
        cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
        cout << endl << e.what() << endl;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;
    }
}
