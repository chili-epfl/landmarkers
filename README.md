# landmarkers

This is a script to test the dlib's face landmarkers detector. We use it to
perform on-line head position recognition with a webcam.

Installation :

    > mkdir build
    > cd build
    > cmake ..
    > ccmake ..

    make sur you are using the good configuration to use your GPU (sse2/sse4) in
    order to optimise the performance of your compurter

    > make

Run :

    you need to be in the directory that contains the file
    shape_predictor_68_face_landmarks.dat (in /share) :

    > cd ../share
    
    then launch the executable :

    > ../build/./head_pos
