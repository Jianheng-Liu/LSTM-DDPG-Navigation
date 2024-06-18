# -*- coding:UTF-8 -*-
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image 

rospy.init_node('Depth Reader')
 
data = None
while data is None:
    try:
        data = rospy.wait_for_message("/kinect2/sd/image_depth_rect", Image, timeout=5)
    except:
        print('Cannot get depth image')

print(data.height)
print(data.width)
print(type(data.data))

cv_img = CvBridge().imgmsg_to_cv2(data, "16UC1")
print(type(cv_img))
print(cv_img)

overall = 0
average = 0
count = 0

center_x = 424 / 2
center_y = 512 / 2
for i in range(256):
    for j in range(256):
        if cv_img[center_x - 128 + i][center_y - 128 + j] > 0:
            overall += cv_img[center_x - 128 + i][center_y - 128 + j] * 0.001
            count += 1

if count > 0:
    average = overall / count

print(overall)
print(average)
print(count)
