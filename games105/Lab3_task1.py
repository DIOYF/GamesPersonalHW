import numpy as np
from scipy.spatial.transform import Rotation as R
from bvh_loader import BVHMotion
from physics_warpper import PhysicsInfo


def part1_cal_torque(pose, physics_info: PhysicsInfo, **kargs):
    '''
    输入： pose： (20,4)的numpy数组，表示每个关节的目标旋转(相对于父关节的)
           physics_info: PhysicsInfo类，包含了当前的物理信息，参见physics_warpper.py
           **kargs：指定参数，可能包含kp,kd
    输出： global_torque: (20,3)的numpy数组，表示每个关节的全局坐标下的目标力矩，根节点力矩会被后续代码无视
    '''
    # ------一些提示代码，你可以随意修改------------#
    # 使用**kargs来指定参数获得需要的参数
    kp = kargs.get('kp', 300) # 需要自行调整kp和kd！ 而且也可以是一个数组，指定每个关节的kp和kd。比例控制和微分控制
    kd = kargs.get('kd', 20)

    parent_index = physics_info.parent_index
    joint_name = physics_info.joint_name
    joint_orientation = physics_info.get_joint_orientation()
    joint_avel = physics_info.get_body_angular_velocity()


    local_orientation = np.zeros((20,3))
    global_torque = np.zeros((20, 3))
    local_orientation[0] = (R.from_quat(pose[0]) * R.from_quat(joint_orientation[0])).as_euler("XYZ",degrees=True)
    for i in range(1, len(joint_orientation)):
        local_orientation[i] = (R.from_quat(pose[i]) * (R.from_quat(joint_orientation[i]) * R.from_quat(joint_orientation[parent_index[i]]).inv()).inv()).as_euler("XYZ",degrees=True)



    local_torque = kp * local_orientation - kd * joint_avel# 相对旋转的微分得到角速度
    for i in range(1, len(global_torque)):
        global_torque[i] = local_torque[i]

    return global_torque

def part2_cal_float_base_torque(target_position, pose, physics_info, **kargs):
    '''
    输入： target_position: (3,)的numpy数组，表示根节点的目标位置，其余同上
    输出： global_root_force: (3,)的numpy数组，表示根节点的全局坐标下的辅助力
          global_torque: 同上
    注意：
        1. 你需要自己计算kp和kd，并且可以通过kargs调整part1中的kp和kd
        2. global_torque[0]在track静止姿态时会被无视，但是track走路时会被加到根节点上，不然无法保持根节点朝向
    '''
    global_torque = part1_cal_torque(pose, physics_info)
    kp = kargs.get('root_kp', 4000) # 需要自行调整root的kp和kd！
    kd = kargs.get('root_kd', 20)
    root_position, root_velocity = physics_info.get_root_pos_and_vel()
    global_root_force = np.zeros((3,))
    # 计算根节点施加外力的大小
    global_root_force = kp * (target_position - root_position) - kd * root_velocity
    return global_root_force, global_torque

def part3_cal_static_standing_torque(bvh: BVHMotion, physics_info):
    '''
    输入： bvh: BVHMotion类，包含了当前的动作信息，参见bvh_loader.py
    其余同上
    Tips: 
        只track第0帧就能保持站立了
        为了保持平衡可以把目标的根节点位置适当前移，比如把根节点位置和左右脚的中点加权平均
        为了仿真稳定最好不要在Toe关节上加额外力矩
    '''
    tar_pos = bvh.joint_position[0][0]
    joint_positions = physics_info.get_joint_translation()
    # 适当前移
    tar_pos = tar_pos * 0.8 + joint_positions[9] * 0.1 + joint_positions[10] * 0.1

    pose = bvh.joint_rotation[0]
    joint_name = physics_info.joint_name
    global_torque = part1_cal_torque(pose, physics_info) # part1 torque
    joint_velocity = physics_info.get_body_velocity()
    joint_mass = physics_info.get_body_mass()
    # compute center of mass and COM velocity
    com = np.zeros(3)
    com_velocity = np.zeros(3)
    mass = 0
    for i in range(len(joint_mass)):
        com += joint_mass[i] * joint_positions[i]
        com_velocity = joint_mass[i] * joint_velocity[i]
        mass += joint_mass[i]
    com /= mass
    com_velocity /= mass
    desired_com = tar_pos

    Kp = 4000
    Kd = 20
    virtual_force = Kp * (desired_com - com) - Kd * com_velocity
    # 做功转化,推导，得到 torque_i = (x - p_i) .cross (f)
    torque = global_torque


    for i in range(0, len(torque)):
        torque[i] -= np.cross( com - joint_positions[i], virtual_force)
    return torque