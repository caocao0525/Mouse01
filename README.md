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

    * if mouse turns to the right (90 degrees), the left eye is invisible. (x, y for the left eye = NaN)
    * if mouse turns to the left (90 degrees), the right eye is invisible. (x, y for the right eye = NaN)
    * if mouse ups its head (90 degrees), both eyes are invisible.
    * if mouse downs its head (90 degrees), all points are visible.



