from yumipy.yumi_robot import YuMiRobot
from yumipy.yumi_state import YuMiState
from autolab_core import RigidTransform
import numpy as np
from yumiplanning.yumi_kinematics import YuMiKinematics as YK
from phoxipy.phoxi_sensor import PhoXiSensor
from grasp import Grasp

'''
This class invokes yumipy, phoxipy, yumiplanning to give a user-friendly interface for
sending motion commands, getting depth camera data, and etc

Consider it an abstract wrapper for the yumi+phoxi which enables some project-specific
capabilities without polluting the original code
(ie keep the yumipy,yumiplanning repo's pure to their intent)
'''
class Interface:
    GRIP_DOWN_R = np.diag([1,-1,-1])#orientation where the gripper is facing downwards
    def __init__(self,phoxi_name,l_tcp,r_tcp,speed=(300,360)):
        #set up the robot
        self.L_TCP=l_tcp
        self.R_TCP=r_tcp
        self.speed=speed
        self.y=YuMiRobot()
        self.yk = YK()
        self.yk.set_tcp(self.L_TCP,self.R_TCP)
        self.y.set_tcp(left=self.L_TCP,right=self.R_TCP)
        self.set_speed(speed)
        #set up the phoxi
        self.T_PHOXI_BASE = RigidTransform.load("/nfs/diskstation/calib/phoxi/phoxi_to_world.tf").as_frames(from_frame="phoxi",to_frame="base_link")
        self.cam = PhoXiSensor("1703005")
        self.cam.start()
        img=self.cam.read()
        self.cam.intrinsics=self.cam.create_intr(img.width,img.height)

    def take_image(self):
        return self.cam.read()
    def set_speed(self,speed):
        '''
        set tcp move speed. format is tuple of (mm/s,deg/s)
        '''
        self.speed=speed
        self.y.set_v(*speed)
    def home(self):
        self.y.left.goto_state(YuMiState(YK.urdf_format_2_yumi(YK.L_NICE_STATE)),wait_for_res=False)
        self.y.right.goto_state(YuMiState(YK.urdf_format_2_yumi(YK.R_NICE_STATE)),wait_for_res=False)
        self.y.left.ping()
        self.y.right.ping()
        
    def __del__(self):
        self.cam.stop()
        self.y.stop()

    def open_grippers(self):
        self.y.left.open_gripper(wait_for_res=False)
        self.y.right.open_gripper(wait_for_res=False)
        self.y.left.ping()
        self.y.right.ping()

    def close_grippers(self):
        self.y.left.close_gripper(wait_for_res=False)
        self.y.right.close_gripper(wait_for_res=False)
        self.y.left.ping()
        self.y.right.ping()

    #move robot to the given point
    def go_poses(self,l_targets=[],r_targets=[],fine=False):
        l_cur_q = self.y.left.get_state().urdf_format
        r_cur_q = self.y.right.get_state().urdf_format
        l_cur_p, r_cur_p = self.yk.fk(qleft=l_cur_q,qright=r_cur_q)
        lpts=[l_cur_p]+l_targets
        rpts=[r_cur_p]+r_targets
        #compute the actual path (THESE ARE IN URDF ORDER (see urdf_order_2_yumi for details))
        lpath,rpath = self.yk.interpolate_cartesian_waypoints(l_waypoints=lpts,l_qinit=l_cur_q,
                    r_qinit=r_cur_q,r_waypoints=rpts,N=20)
        self.go_configs(l_q=lpath,r_q=rpath)
        #this "fine" motion is to account for the fact that our ik is slightly (~.5cm) wrong becase
        # the urdf file is wrong. justin is actively looking into this....
        if fine:
            if(len(l_targets)>0):self.y.left.goto_pose(l_targets[0],relative=True)
            if(len(r_targets)>0):self.y.right.goto_pose(r_targets[0],relative=True)

    def go_configs(self,l_q=[],r_q=[]):
        '''
        moves the arms along the given joint trajectories
        l_q and r_q should be given in urdf format as a np array
        '''
        if(isinstance(l_q,YuMiState) or isinstance(r_q,YuMiState)):
            raise Exception("go_configs takes in np arrays, not YuMiStates")
        for i in range(len(l_q)):
            self.y.left.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(l_q[i])))
        for i in range(len(r_q)):
            self.y.right.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(r_q[i])))
        self.y.left.joint_buffer_execute(wait_for_res=False)
        self.y.right.joint_buffer_execute()
        self.y.right.joint_buffer_clear()
        self.y.left.joint_buffer_clear()#this final one is like a sync on the arms moving together
    def grasp(self,l_grasp=None,r_grasp=None):
        '''
        Carries out the grasp motions on the desired end poses. responsible for doing the 
        approach to pre-grasp as well as the actual grasp itself.
        attempts
        both arguments should be a Grasp object
        '''
        l_waypoints=[]
        r_waypoints=[]
        self.y.left.close_gripper(no_wait=True)
        self.y.right.close_gripper(no_wait=True)
        if l_grasp is not None:
            pre=l_grasp.compute_pregrasp()
            l_waypoints.append(pre)
        if r_grasp is not None:
            pre=r_grasp.compute_pregrasp()
            r_waypoints.append(pre)
        self.go_poses(l_waypoints,r_waypoints,fine=True)
        #move in
        self.sync()
        if l_grasp is not None:
            self.y.left.set_speed(l_grasp.speed)
            self.y.left.goto_pose(l_grasp.compute_gripopen(),relative=True,linear=True,wait_for_res=False)
        if r_grasp is not None:
            self.y.right.set_speed(r_grasp.speed)
            self.y.right.goto_pose(r_grasp.compute_gripopen(),relative=True,linear=True,wait_for_res=False)
        #put grippers in right position
        self.sync()
        if l_grasp is not None:
            self.y.left.move_gripper(l_grasp.gripper_pos,wait_for_res=False)
        if r_grasp is not None:
            self.y.right.move_gripper(r_grasp.gripper_pos,wait_for_res=False)
        self.sync()
        if l_grasp is not None:
            self.y.left.goto_pose(l_grasp.pose,relative=True,linear=True,wait_for_res=False)
        if r_grasp is not None:
            self.y.right.goto_pose(r_grasp.pose,relative=True,linear=True,wait_for_res=False)
        self.sync()
        #grasp
        if l_grasp is not None:
            self.y.left.close_gripper(wait_for_res=False)
        if r_grasp is not None:
            self.y.right.close_gripper(wait_for_res=False)
        self.sync()
        self.set_speed(self.speed)
    def sync(self):
        self.y.left.ping()
        self.y.right.ping()
    def refine_states(self,left=True,right=True,t_tol=.05,r_tol=.3):
        '''
        attempts to move the arms into a better joint configuration without deviating much at the end effector pose
        '''
        l_q = self.y.left.get_state().urdf_format
        r_q = self.y.right.get_state().urdf_format
        l_traj,r_traj = self.yk.refine_state(l_state=l_q,r_state=r_q,t_tol=t_tol,r_tol=r_tol)
        if not left:l_traj=[]
        if not right:r_traj=[]
        self.go_configs(l_q=l_traj,r_q=r_traj)

    def shake_left(self,rot,num_shakes):
        '''
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        '''
        #assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp=self.yk.l_tcp
        old_r_tcp=self.yk.r_tcp
        rot=np.array(rot)
        self.yk.set_tcp(None,None)
        self.y.set_tcp(None,None)
        cur_state = self.y.left.get_state().urdf_format
        curpose,_ = self.yk.fk(qleft=cur_state)
        R_for = RigidTransform(rotation=RigidTransform.rotation_from_axis_angle(rot/2.0),
            from_frame=self.yk.l_tcp_frame,to_frame=self.yk.l_tcp_frame)
        R_back = RigidTransform(rotation=RigidTransform.rotation_from_axis_angle(-rot/2.0),
            from_frame=self.yk.l_tcp_frame,to_frame=self.yk.l_tcp_frame)
        target_for = curpose*R_for
        target_back = curpose*R_back
        target_for_q,_=self.yk.ik(left_qinit=cur_state,left_pose=target_for)
        target_back_q,_=self.yk.ik(left_qinit=cur_state,left_pose=target_back)
        if(np.linalg.norm(target_for_q-target_back_q)>3):
            print("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp,old_r_tcp)
            self.y.set_tcp(left=old_l_tcp,right=old_r_tcp)
            return
        self.y.left.joint_buffer_clear()
        for i in range(num_shakes):
            self.y.left.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(target_for_q)))
            self.y.left.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(target_back_q)))
        self.y.left.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(cur_state)))
        self.y.left.joint_buffer_execute()
        self.y.left.joint_buffer_clear()
        self.yk.set_tcp(old_l_tcp,old_r_tcp)
        self.y.set_tcp(left=old_l_tcp,right=old_r_tcp)
    def shake_right(self,rot,num_shakes):
        '''
        Shake the gripper by translating by trans and rotating by rot about the wrist
        rot is a 3-vector in axis-angle form (the magnitude is the amount)
        trans is a 3-vector xyz translation
        arms is a list containing the YuMiArm objects to shake (for both, pass in both left and right)

        '''
        #assumed that translation is small so we can just toggle back and forth between poses
        old_l_tcp=self.yk.l_tcp
        old_r_tcp=self.yk.r_tcp
        rot=np.array(rot)
        self.yk.set_tcp(None,None)
        self.y.set_tcp(None,None)
        cur_state = self.y.right.get_state().urdf_format
        _,curpose = self.yk.fk(qright=cur_state)
        R_for = RigidTransform(rotation=RigidTransform.rotation_from_axis_angle(rot/2.0),
            from_frame=self.yk.r_tcp_frame,to_frame=self.yk.r_tcp_frame)
        R_back = RigidTransform(rotation=RigidTransform.rotation_from_axis_angle(-rot/2.0),
            from_frame=self.yk.r_tcp_frame,to_frame=self.yk.r_tcp_frame)
        target_for = curpose*R_for
        target_back = curpose*R_back
        _,target_for_q=self.yk.ik(right_qinit=cur_state,right_pose=target_for)
        _,target_back_q=self.yk.ik(right_qinit=cur_state,right_pose=target_back)
        if(np.linalg.norm(target_for_q-target_back_q)>3):
            print("aborting shake action, no smooth IK found between shake poses")
            self.yk.set_tcp(old_l_tcp,old_r_tcp)
            self.y.set_tcp(left=old_l_tcp,right=old_r_tcp)
            return
        self.y.right.joint_buffer_clear()
        for i in range(num_shakes):
            self.y.right.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(target_for_q)))
            self.y.right.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(target_back_q)))
        self.y.right.joint_buffer_add(YuMiState(YK.urdf_format_2_yumi(cur_state)))
        self.y.right.joint_buffer_execute()
        self.y.right.joint_buffer_clear()
        self.yk.set_tcp(old_l_tcp,old_r_tcp)
        self.y.set_tcp(left=old_l_tcp,right=old_r_tcp)
