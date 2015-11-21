Sample camera calibration code.  Calibration part of this code is from
OpenCV python code samples.  You will have to supply your own calibration
chessboard images.  You will also have to plug in your own square_size
and pattern_size parameters (my sample chessboard images have 54 corner
points)
I added two methods - one to calculate image points, given world points
(using projectPoints method of opencv) and world points, given image
points - unsing homography
