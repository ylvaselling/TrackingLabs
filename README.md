# Tracking with Lucas-Kanade-tracker and multiscale optical flow
In these labs, we have implemented algorithms for finding  Harris features (intrinsically 2D points) and to track them with the Lucas-Kanade algorithm. It is written in python and uses the numpy library.

## Harris feature points
![Harris feature points](https://raw.githubusercontent.com/ylvaselling/TrackingLabs/master/img/Harris_features.png)
## Visualization of optical flow, zooming motion
This calculation of optical flow is done by the multiscale Lucas-Kanade algorithm and can be seen here on a zooming optical flow. Therefore, we can see in the resulting image a circle-like feature, with different colors around it. The colors represent vector directions and the vectors are pointing to the middle of the image, where the direction of the optical flow is headed.
### Image 1
![Image 1](https://raw.githubusercontent.com/ylvaselling/TrackingLabs/master/img/forwardL0.png)
### Image 2
![Image 2](https://raw.githubusercontent.com/ylvaselling/TrackingLabs/master/img/forwardL1.png)
## Result, optical flow  ![Visualization of displacement](https://raw.githubusercontent.com/ylvaselling/TrackingLabs/master/img/Displacement_zoom.png)
