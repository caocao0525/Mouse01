# Mouse01

### Mouse pose determinator

**Source File**
In csv format, contain the 2D coordinates
columns are :
img_no, ear_r_x, ear_r_y, ear_l_x, ear_l_y, eye_r_x, eye_r_y, eye_l_x, eye_l_y, nose_x, nose_y

total observation: 4555

The image size = 400x300 (width: 400, height:300)

**Direction estimate**

*hypothesis*: 5 classes for pan angle and tilt angle, respectively.

*rough estimation*

   >* if mouse turns to the right (90 degrees), the left eye is invisible. (x, y for the left eye = NaN)
   >* if mouse turns to the left (90 degrees), the right eye is invisible. (x, y for the right eye = NaN)
   >* if mouse ups its head (90 degrees), both eyes are invisible.
   >* if mouse downs its head (90 degrees), all points are visible.

**Exceptions**
In some cases, points might not be detected due to the position of the mouse head (too left, too right).

**X-axis**
For the most straight-forward direction, the x coordiates of each point should be :
 R_e < R_i < N < L_i < L_e (where e stands for ear and i indicates eye.)
 
**For pan angles of +/- 45 degrees**
   >* estimation for 45 degrees to the right: R_e ~ R_i ~ N < L_i < L_e
   >* estimation for 45 degrees to the left:  R_e < R_i < N ~ L_i ~ L_e

Seems like any norma for the straight forward direction is required, in terms of ratio of each detected body point.