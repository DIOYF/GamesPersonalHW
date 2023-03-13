# 以下部分均为可更改部分

from answer_task1 import *
from smooth_utils import quat_to_avel, decay_spring_implicit_damping_pos,decay_spring_implicit_damping_rot

class CharacterController():
    def __init__(self, controller) -> None:
        self.motions = []
        self.motions.append(BVHMotion('motion_material/walk_forward.bvh'))  # 100
        self.motions.append(BVHMotion('motion_material/idle.bvh'))
        self.motion_id = 0
        self.controller = controller
        self.cur_root_pos = None
        self.cur_root_rot = None
        self.cur_frame = 0
        self.blending_motion = build_loop_motion(self.motions[0])
        self.idle_motion = build_loop_motion(self.motions[1])
        self.idle2move_motion = concatenate_two_motions(self.motions[1], self.motions[0], 60, 30)
        self.move2idle_motion = concatenate_two_motions(self.motions[0], self.motions[1], 60, 30)
        self.motion_state = "idle"


        pass
    
    def update_state(self, 
                     desired_pos_list, 
                     desired_rot_list,
                     desired_vel_list,
                     desired_avel_list,
                     current_gait
                     ):
        '''
        此接口会被用于获取新的期望状态
        Input: 平滑过的手柄输入,包含了现在(第0帧)和未来20,40,60,80,100帧的期望状态,以及一个额外输入的步态
        简单起见你可以先忽略步态输入,它是用来控制走路还是跑步的
            desired_pos_list: 期望位置, 6x3的矩阵, 每一行对应0，20，40...帧的期望位置(水平)， 期望位置可以用来拟合根节点位置也可以是质心位置或其他
            desired_rot_list: 期望旋转, 6x4的矩阵, 每一行对应0，20，40...帧的期望旋转(水平), 期望旋转可以用来拟合根节点旋转也可以是其他
            desired_vel_list: 期望速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望速度(水平), 期望速度可以用来拟合根节点速度也可以是其他
            desired_avel_list: 期望角速度, 6x3的矩阵, 每一行对应0，20，40...帧的期望角速度(水平), 期望角速度可以用来拟合根节点角速度也可以是其他
        
        Output: 同作业一,输出下一帧的关节名字,关节位置,关节旋转
            joint_name: List[str], 代表了所有关节的名字
            joint_translation: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
            joint_orientation: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
        Tips:
            输出三者顺序需要对应
            controller 本身有一个move_speed属性,是形状(3,)的ndarray,
            分别对应着面朝向移动速度,侧向移动速度和向后移动速度.目前根据LAFAN的统计数据设为(1.75,1.5,1.25)
            如果和你的角色动作速度对不上,你可以在init或这里对属性进行修改
        '''
        # 一个简单的例子，输出第i帧的状态
        # 只需要根据simulation bone完成从motion_origin到motion_target的转换
        # 根据desired_pos_list , 判断当前motion_id和目标motion_id， 将两者进行blending
        # motion_id : 0 move , 1 idle
        joint_name = self.blending_motion.joint_name

        last_motion_state = self.motion_state
        self.motion_state = "idle" if abs(desired_vel_list[0,0])+abs(desired_vel_list[0,1]) < 1e-2 else "move"

        if self.motion_state == "move":
            motion_id = self.motion_id
            current_motion = self.blending_motion.raw_copy()
            if self.motion_state != last_motion_state:
                facing_axis = R.from_quat(self.idle_motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                current_motion = current_motion.translation_and_rotation(0, self.idle_motion.joint_position[self.cur_frame, 0, [0, 2]],
                                                                         facing_axis)



                self.cur_frame = 0
            key_frame = [(self.cur_frame + 20 * i) % self.motions[motion_id].motion_length for i in range(6)]
            current_motion_key_frame_vel = current_motion.joint_position[key_frame, 0, :] - current_motion.joint_position[[(frame - 1) for frame in key_frame], 0, :]
            current_motion_avel = quat_to_avel(current_motion.joint_rotation[:, 0, :], 1 / 60)

            # It is only for root bone

            diff_root_pos = desired_pos_list - current_motion.joint_position[ key_frame, 0, :]
            diff_root_pos[:, 1] = 0
            diff_root_rot = (R.from_quat(desired_rot_list[0:6]) * R.from_quat(current_motion.joint_rotation[ key_frame, 0, :]).inv()).as_rotvec()
            diff_root_vel = (desired_vel_list - current_motion_key_frame_vel)/60
            diff_root_avel = desired_avel_list[0:6] - current_motion_avel[[(frame-1) for frame in key_frame]]

            for i in range(self.cur_frame, self.cur_frame+self.motions[motion_id].motion_length//2):
                half_time = 0.2
                index = (i - self.cur_frame) // 20
                dt = (i-self.cur_frame) % 20

                off_pos, _ = decay_spring_implicit_damping_pos(diff_root_pos[index], diff_root_vel[index], half_time, dt/60)
                off_rot, _ = decay_spring_implicit_damping_rot(diff_root_rot[index], diff_root_avel[index], half_time, dt/60)

                current_motion.joint_position[ i % self.motions[motion_id].motion_length, 0, :] += off_pos
                current_motion.joint_rotation[ i % self.motions[motion_id].motion_length, 0, :] = (R.from_rotvec(off_rot) * R.from_quat(current_motion.joint_rotation[ i % self.motions[motion_id].motion_length, 0, :])).as_quat()

            joint_translation, joint_orientation = current_motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]

            self.blending_motion = current_motion
            self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

        elif self.motion_state == "idle":
            motion_id = self.motion_id
            current_motion = self.idle_motion
            if self.motion_state != last_motion_state:
                facing_axis = R.from_quat(self.blending_motion.joint_rotation[self.cur_frame, 0, :]).apply(np.array([0, 0, 1])).flatten()[[0, 2]]
                current_motion = current_motion.translation_and_rotation(0, self.blending_motion.joint_position[self.cur_frame, 0, [0, 2]],
                                                                         facing_axis)
                self.cur_frame = 0

            joint_translation, joint_orientation = current_motion.batch_forward_kinematics()
            joint_translation = joint_translation[self.cur_frame]
            joint_orientation = joint_orientation[self.cur_frame]
            self.cur_root_pos = joint_translation[0]
            self.cur_root_rot = joint_orientation[0]
            self.cur_frame = 0
            self.idle_motion = current_motion
            self.cur_frame = (self.cur_frame + 1) % self.motions[motion_id].motion_length

        # 返回的只是一帧的结果
        return joint_name, joint_translation, joint_orientation
    

    def sync_controller_and_character(self, controller, character_state):
        '''
        这一部分用于同步你的角色和手柄的状态
        更新后很有可能会出现手柄和角色的位置不一致，这里可以用于修正
        让手柄位置服从你的角色? 让角色位置服从手柄? 或者插值折中一下?
        需要你进行取舍
        Input: 手柄对象，角色状态
        手柄对象我们提供了set_pos和set_rot接口,输入分别是3维向量和四元数,会提取水平分量来设置手柄的位置和旋转
        角色状态实际上是一个tuple, (joint_name, joint_translation, joint_orientation),为你在update_state中返回的三个值
        你可以更新他们,并返回一个新的角色状态
        '''
        
        # 一个简单的例子，将手柄的位置与角色对齐
        controller.set_pos(self.cur_root_pos)
        controller.set_rot(self.cur_root_rot)

        return character_state


