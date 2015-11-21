#!/usr/bin/python

import os

import sys

from glob import glob

import numpy as np

import cv2

import matplotlib.pyplot as plt


# Camera calibration code from python opencv samples
# Of particular interest is if I can map pixels from
# pixel space back to world coordinates and if those
# coordinate values are correct

# Some questions to ponder:
# 1.  What is the role of Z?  Is there a Z estimate for
#     the world coordinates?  Likely, just the T vector?
# 2.  Do you really need the Z value?  Can you correlate
#     from pixel space to world space where Z is 0 by
#     default (i.e. world coordinates are computed on the
#     plane of checkerboard only) - Yes, this is homography
#     The imgToWorld method uses this to map image points
#     back to world space.  The output is a 3-vector with
#     the third element being a scale factor (first and second
#     are (X,Y)
# 3.  if [x,y,1] = [K][R][X,Y,Z] + t, then
#     [X,Y,Z] should be derivable by applying inv transforms
#     Not quite - unless your board coordinates presume zero Z
#     in which case RT matrix becomes 3x3 and [K][RT] is 3x3
#     and invertible


class cam_cal(object):

    def __init__(self, fname_mask):
        self.fname_mask = fname_mask

        # Distortion parameters matrix
        self.distort = None

        # Camera intrinsics, converting focal lenght
        # to pixels offset from top left
        self.intrinsic = None

        self.img_points = []

        self.obj_points = []

        self.rvecs = None

        self.tvecs = None

        # Parameters that should really be cmd line options
        self.square_size = 1.0

        self.pattern_size = (9, 6)


    def camCalibrate(self):
        '''
        Calibrate camera - get camera distortion and intrinsic
        parameters - fixed for a fixed focal length
        (Attribbution: This code picked from opencv sample python
        programs)
        '''
        fnames = glob(self.fname_mask)

        pattern_points = np.zeros((np.prod(self.pattern_size), 3), np.float32)

        pattern_points[:,:2] = np.indices(self.pattern_size).T.reshape(-1,2)

        pattern_points *= self.square_size

        h, w = 0, 0

        view_count = 0

        for fn in fnames:
            print 'processing %s...' % fn,

            img = cv2.imread(fn, 0)

            if img is None:
                print "Failed to load", fn

                continue

            h, w = img.shape[:2]

            found, corners = cv2.findChessboardCorners(img, self.pattern_size)

            if found:
                # [TODO] Understand these parameters
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )

                # [TODO] Understand these parameters
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            else:
                print 'chessboard not found'

                continue

            self.img_points.append(corners.reshape(-1, 2))

            self.obj_points.append(pattern_points)

            view_count += 1

            print 'ok'

        rms, self.intrinsic, self.distort, self.rvecs, self.tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, (w, h), None, None)

        print "RMS:", rms

        print "camera matrix:\n", self.intrinsic

        print "distortion coefficients: ", self.distort.ravel()


    def undistortViews(self):
        '''
        This is optional - first call initUndistortRectifyMap, to get
        a rectification map for all camera images (views), then call
        the remap function to undistort each view, then call
        findBoardCorners again, to compute adjusted corners
        '''
        pass


    def imgToWorld(self, view, point):
        '''
        Get RT matrix for image, then take img_pt and
        apply inverse transform to get world coords
        [TODO] This does not work - not sure why - need to
        investigate
        '''
        print 'view: %d;  point: %d' % (view, point)

        rmat, _ = cv2.Rodrigues(self.rvecs[view])

        img_point = np.matrix(np.append(self.img_points[view][point], 1))

        img_point = img_point.T

        print 'img_point: ', img_point

        irmat = np.linalg.inv(rmat)

        iintr = np.linalg.inv(self.intrinsic)

        p = iintr * img_point

        print 'iintr * img_point - p:', p, 'shape: ', p.shape

        print 'tvec: ', self.tvecs[view], 'shape: ', self.tvecs[view].shape

        tvec = np.matrix(self.tvecs[view])

        # out = irmat * p - irmat * self.tvecs[0]

        out = irmat * p

        print 'irmat * p: ', out

        out2 = out - irmat * tvec

        print 'irmat * p - irmat * tvec - ', out2

        print 'target board point: ' , self.obj_points[view][point]


    def imgToWorld2(self, view, point):
        '''
        Set Z to zero, then middle column of rotaation matrix is 0 and we get
        a 3x3 extrinsic matrix (homogrphy).  Then apply inverse
        '''
        print 'view: %d;  point: %d' % (view, point)

        rmat, _ = cv2.Rodrigues(self.rvecs[view])

        print 'rmat: ', rmat

        img_point = np.matrix(np.append(self.img_points[view][point], 1))

        img_point = img_point.T

        rmat_noZ = rmat[:, :2]

        print 'rmat_noZ: ', rmat_noZ

        tvec = np.matrix(self.tvecs[view])

        homog = np.column_stack((rmat_noZ, tvec))

        print 'homog: ', homog

        homog = self.intrinsic * homog

        ihomog = np.linalg.inv(homog)

        out = ihomog * img_point

        out /= out[2]

        print 'out: ', out

        print 'target board point: ', self.obj_points[view][point]

 

    def worldToImg(self):
        '''
        Get RT matrix for image, then take world coords
        and apply intrinsic and extrinsic transforms to
        get image point
        '''
        img_points, _ = cv2.projectPoints(self.obj_points[0], self.rvecs[0], self.tvecs[0], self.intrinsic, self.distort)

        diff = img_points - self.img_points[0]

        print 'Computed projection:'

        print img_points

        print 'Original projections:'

        print self.img_points[0]

        print  diff


def main():
    '''
    '''
    cc = cam_cal('/home/vijaykam/Downloads/opencv-3.0.0/samples/data/left*.jpg')

    cc.camCalibrate( )

    # cc.worldToImg( )

    # cc.imgToWorld(0,48)

    cc.imgToWorld2(0,48)

 
if __name__ == '__main__':
    main()

