import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    # file read
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

        joint_name = []
        joint_parent = []
        joint_offset = []
        stack = []  # using stack to track parent index for each joint
        # it is a tree seq question
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                joint_name.append(lines[i].split()[1])
                joint_parent.append(-1)
            elif lines[i].startswith('MOTION'):
                break
            else:
                tmp_line = lines[i].split()
                if tmp_line[0] == '{':
                    stack.append(len(joint_name)-1)  # parent index is joint_name[-1]
                elif tmp_line[0] == '}':
                    stack.pop()
                elif tmp_line[0] == 'JOINT':
                    joint_name.append(tmp_line[1])
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == 'End':  # align push and pop operation
                    joint_name.append(joint_name[stack[-1]]+'_end')
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == 'OFFSET':
                    joint_offset.append(np.array([float(x) for x in tmp_line[1:4]]).reshape(1, -1))
                else:
                    continue

    joint_offset = np.concatenate(joint_offset, axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
    """
    # Transition data,  about channels position and channels rotations to index joint
    motion_channels_data = motion_data[frame_id]
    root_position = np.array(motion_channels_data[0:3])
    joint_local_rotation = []
    count = 0
    for i in range(len(joint_name)):
        if '_end' in joint_name[i]:
            joint_local_rotation.append([0., 0., 0.])
        else:
            joint_local_rotation.append(motion_channels_data[3*count+3: 3*count+6])
            count += 1

    # Traverse list, parent node compute R finished before child node, so traverse list from start to end is OK
    joint_positions = []
    joint_orientations = []
    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            # Qroot = bvh_channels_get_root_rotation
            # Proot = bvh_channels_get_root_position
            joint_orientation = R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_position = root_position.reshape(1, -1)  # align matrix dimension
        else:
            # Qi = Qi-parent * bvh_channels_get_i_rotation
            # Pi = Pi-parent + offset-i * Oi-parent.T ,  note: Raw Vector * transpose Right Rotation matrix
            joint_orientation = R.from_quat(joint_orientations[joint_parent[i]][0]) * R.from_euler('XYZ', joint_local_rotation[i], degrees=True)
            joint_position = joint_positions[joint_parent[i]] + joint_offset[i] * np.asmatrix(R.from_quat(joint_orientations[joint_parent[i]][0]).as_matrix()).transpose()
        joint_positions.append(np.array(joint_position))
        joint_orientations.append(joint_orientation.as_quat().reshape(1, -1))

    joint_positions = np.concatenate(joint_positions, axis=0)
    joint_orientations = np.concatenate(joint_orientations, axis=0)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
    """
    # T_joint_name , get A_joint_name for index with A_motion data
    T_joint_name, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, _, _ = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)

    # get a hash index for motion data c.s.t joint_name
    A_joint_map = {}
    count = 0
    for i in range(len(A_joint_name)):
        if '_end' in A_joint_name[i]:
            count += 1
        A_joint_map[A_joint_name[i]] = i - count

    motion_data = []
    # for i in range(1): debug init pose. lShoulder add 0,0,-45, rShoulder add 0,0,45
    for i in range(A_motion_data.shape[0]):
        data = []
        for joint in T_joint_name:
            index = A_joint_map[joint]

            if joint == 'RootJoint':
                data += list(A_motion_data[i][0:6])
            elif joint == 'lShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., -45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
            elif joint == 'rShoulder':
                Rot = (R.from_euler('XYZ', list(A_motion_data[i][index * 3 + 3: index*3 + 6]), degrees=True) * R.from_euler('XYZ', [0., 0., 45.], degrees=True)).as_euler('XYZ',True)
                data += list(Rot)
            elif '_end' in joint:
                continue
            else:
                data += list(A_motion_data[i][index * 3 + 3: index * 3 + 6])
        motion_data.append(np.array(data).reshape(1, -1))

    motion_data = np.concatenate(motion_data, axis=0)
    return motion_data

