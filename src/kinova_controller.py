#!/usr/bin/env python
import math
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from kinova_msgs.msg import JointTorque
from kinova_msgs.srv import Stop ,Start
import actionlib
import kinova_msgs.msg
import time 

class KinovaIKController:
    def __init__(self):
        rospy.init_node('kinova_ik_controller')      
        # ROS通信
        self.joint_state_sub = rospy.Subscriber('teleoperation_joint_state', JointState, self.joint_state_callback, queue_size=1)

        action_address = '/j2n6s300_driver/joints_action/joint_angles'
        self.client = actionlib.SimpleActionClient(action_address,
                                          kinova_msgs.msg.ArmJointAnglesAction)
        self.client.wait_for_server()
        # self.stop_service = rospy.ServiceProxy('/j2n6s300_driver/in/stop', Stop)
        # self.start_service = rospy.ServiceProxy('/j2n6s300_driver/in/start', Start)
        # self.q_target = np.array([np.pi,np.pi,np.pi,np.pi,np.pi,np.pi])
        self.q_target = None
        print("init success")

    def joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.q_target = [pos for _ , pos in zip(msg.name, msg.position)]
        print(f'get joint state: {self.q_target}')
        # self._publish_joint_command(q_target)
    
    def _publish_joint_command(self, q_target):
        """发布关节指令"""
        if self.q_target is None:
            return 
        rospy.loginfo(f"receive command")
        goal = kinova_msgs.msg.ArmJointAnglesGoal()
        goal.angles.joint1 = q_target[0]
        goal.angles.joint2 = q_target[1]
        goal.angles.joint3 = q_target[2]
        goal.angles.joint4 = q_target[3]
        goal.angles.joint5 = q_target[4]
        goal.angles.joint6 = q_target[5]
        

        self.client.send_goal(goal)
        if self.client.wait_for_result(rospy.Duration(20.0)):
            return self.client.get_result()
        else:
            print(' the joint angle action timed-out')
            self.client.cancel_all_goals()
            return None

    def run(self):
        # rospy.spin()
        rate = rospy.Rate(60)
        while  not rospy.is_shutdown():
            # 这里可以添加其他逻辑
            # self.stop_service()
            # self.start_service()
            # self.client.cancel_all_goals()
            if self.q_target is not None:
                self.client.cancel_all_goals()
                self._publish_joint_command(self.q_target)

            # rate.sleep()
            

if __name__ == '__main__':
    try:
        controller = KinovaIKController()
        controller.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("节点已终止")