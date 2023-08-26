#!/usr/bin/env python3
""" This script is the main node launcher for publishing dgp data.

This script has to run in python3, because dgp only runs in python3.6/python3.7.

One important workaround of the current file is to publish pose instead of tf, Because the installation of tf in python3 is difficult and can pollute the system environments.
We take a detour and publish PoseStamped instead, and we add another python2 compatible script 'pose2tf.py' to transfer poses into a tf tree.

To cleanly allow python3 import rospy (this should not affect python2):

```bash
sudo apt-get install python3-catkin-pkg-modules
sudo apt-get install python3-rospkg-modules
```

"""
import os
import rospy 
import numpy as np
import cv2
from dgp.datasets.synchronized_dataset import SynchronizedScene
import json
from utils import ros_util
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Int32, Bool, Float32
from geometry_msgs.msg import PoseStamped
import geometry_msgs
import tf

class DgpVisualizationNode:
    def __init__(self):
        rospy.init_node("DgpVisualizationNode")
        rospy.loginfo("Starting DgpVisualizationNode.")

        self.read_params()
        
        # Most Publishers will be initialized right before the first published data
        self.publishers = {}

        self.camera_names = ['CAMERA_01', 'CAMERA_05', 'CAMERA_06', 'CAMERA_07', 'CAMERA_08', 'CAMERA_09']
        self.lidar_name = 'LIDAR'
        # This takes some time and block this process.
        scenes_files = json.load(open(self.ddad_json_path, 'r'))['scene_splits']['0']['filenames']
        base_dir_name = os.path.dirname(self.ddad_json_path)
        self.datasets = [
            SynchronizedScene(
                scene_json=os.path.join(base_dir_name, scene_file),
                datum_names = [self.lidar_name] + self.camera_names
            ) for scene_file in scenes_files
        ]
        rospy.loginfo(len(self.datasets)) # 850 with default data

        # Initialize for control status
        self.set_index(0)
        self.pause = False
        self.stop = True
        self.publishing = True
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_frequency), self.publish_callback)
        
        rospy.Subscriber("/dgp/control/index", Int32, self.index_callback)
        rospy.Subscriber("/dgp/control/stop", Bool, self.stop_callback)
        rospy.Subscriber("/dgp/control/pause", Bool, self.pause_callback)
        
    def read_params(self):
        self.ddad_json_path = rospy.get_param("~DDAD_JSON_PATH", '/data/ddad_train_val/ddad.json')
        self.update_frequency = float(rospy.get_param("~UPDATE_FREQUENCY", 8.0))

    def set_index(self, index):
        """Set current index -> select scenes -> print(scene description) -> set current sample as the first sample of the scene
        """
        self.index = index
        self.float_index = 0
        print(f"Switch to scenes {self.index}")

    def stop_callback(self, msg):
        self.published=False
        self.stop = msg.data #Bool
        self.float_index = 0
        self.publish_callback(None)

    def pause_callback(self, msg):
        self.pause = msg.data

    def index_callback(self, msg):
        self.set_index(msg.data)
        self.publish_callback(None)

    def publish_tf(self, pose_stamped, current_frame, frame_id = None, new_stamp=False):
        assert isinstance(pose_stamped, PoseStamped)
        br = tf.TransformBroadcaster()

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = pose_stamped.header.frame_id if frame_id is None else frame_id
        t.header.stamp = rospy.Time.now() if new_stamp else pose_stamped.header.stamp
        t.child_frame_id = current_frame
        t.transform.translation = pose_stamped.pose.position
        t.transform.rotation    = pose_stamped.pose.orientation

        br.sendTransformMessage(t)

    def _camera_publish(self, camera_data, is_publish_image=False):
        """Publish camera related data, first publish pose/tf information, then publish image if needed

        Args:
            camera_data (Dict): camera data from dgp' sample data
            is_publish_image (bool, optional): determine whether to read image data. Defaults to False.
        """        

        channel = camera_data['datum_name']
        
        # publish relative pose
        cam_intrinsic = camera_data['intrinsics'] #[3 * 3]
        rotation = camera_data['extrinsics'].quat.q #list, [4] r, x, y, z
        translation = camera_data['extrinsics'].tvec #
        relative_pose = ros_util.compute_pose(translation, rotation)
        pose_pub_name = f"{channel}_pose_pub"
        if pose_pub_name not in self.publishers:
            self.publishers[pose_pub_name] = rospy.Publisher(f"/dgp/{channel}/pose", PoseStamped, queue_size=1, latch=True)
        msg = PoseStamped()
        msg.pose = relative_pose
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        self.publishers[pose_pub_name].publish(msg)
        self.publish_tf(msg, channel)    

        if is_publish_image:
            image_pub_name = f"{channel}_image_pub"
            if image_pub_name not in self.publishers:
                self.publishers[image_pub_name] = rospy.Publisher(f"/dgp/{channel}/image", Image, queue_size=1, latch=1)
            info_pub_name  = f"{channel}_info_pub"
            if info_pub_name not in self.publishers:
                self.publishers[info_pub_name] = rospy.Publisher(f"/dgp/{channel}/camera_info", CameraInfo, queue_size=1, latch=1)

            image = cv2.cvtColor(np.array(camera_data['rgb']), cv2.COLOR_RGB2BGR)
            ros_util.publish_image(image,
                                   self.publishers[image_pub_name],
                                   self.publishers[info_pub_name],
                                   cam_intrinsic,
                                   channel)

    def _lidar_publish(self, lidar_data, is_publish_lidar=False):
        """Publish lidar related data, first publish pose/tf information and ego pose, then publish lidar if needed

        Args:
            lidar_data (Dict): lidar data from dgp' sample data
            is_publish_lidar (bool, optional): determine whether to read lidar data. Defaults to False.
        """
        channel = lidar_data['datum_name']
        
        # publish relative pose
        rotation = lidar_data['extrinsics'].quat.q #list, [4] r, x, y, z
        translation = lidar_data['extrinsics'].tvec #
        relative_pose = ros_util.compute_pose(translation, rotation)
        pose_pub_name = f"{channel}_pose_pub"
        if pose_pub_name not in self.publishers:
            self.publishers[pose_pub_name] = rospy.Publisher(f"/dgp/{channel}/pose", PoseStamped, queue_size=1, latch=True)
        msg = PoseStamped()
        msg.pose = relative_pose
        msg.header.frame_id = "base_link"
        msg.header.stamp = rospy.Time.now()
        self.publishers[pose_pub_name].publish(msg)
        self.publish_tf(msg, channel)

        # publish ego pose
        ego_rotation = lidar_data['pose'].quat.q
        ego_translation = lidar_data['pose'].tvec
        frame_location = ros_util.compute_pose(ego_translation, ego_rotation)
        pose_pub_name = "ego_pose"
        
        if pose_pub_name not in self.publishers:
            self.publishers[pose_pub_name] = rospy.Publisher("/dgp/ego_pose", PoseStamped, queue_size=1, latch=True)
        msg = PoseStamped()
        msg.pose = frame_location
        msg.header.frame_id = "world"
        msg.header.stamp = rospy.Time.now()
        self.publishers[pose_pub_name].publish(msg)
        self.publish_tf(msg, pose_pub_name)

        # publish lidar
        if is_publish_lidar:
            point_cloud = np.concatenate([lidar_data['point_cloud'], lidar_data['extra_channels']], axis=-1) #[N, 4]
            lidar_pub_name = 'lidar_pub'
            if lidar_pub_name not in self.publishers:
                self.publishers[lidar_pub_name] = rospy.Publisher(f"/dgp/{channel}/data", PointCloud2, queue_size=1, latch=True)

            ros_util.publish_point_cloud(point_cloud, self.publishers[lidar_pub_name], channel)
              
    def publish_callback(self, event):
        if self.stop: # if stopped, falls back to an empty loop
            return

        data_collected = self.datasets[self.index][self.float_index][0]
        for data in data_collected:
            if data['datum_name'] == self.lidar_name:
                self._lidar_publish(data, is_publish_lidar=self.publishing)
            else:
                self._camera_publish(data, is_publish_image=self.publishing)

        self.publishing = not self.pause # if paused, the original images and lidar are latched (as defined in publishers) and we will not re-publish them to save memory access. But we need to re-publish tf and markers

        if not self.pause:
            self.float_index = (self.float_index + 1) % len(self.datasets[self.index])
            if self.float_index == 0:
                rospy.loginfo("We have reached the end of the dataset, restarting from the beginning")

if __name__ == "__main__":
    ros_node = DgpVisualizationNode()
    rospy.spin()