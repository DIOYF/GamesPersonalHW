import fractions

import future.builtins
import numpy as np

from bvh_utils import *
#---------------你的代码------------------#
# translation 和 orientation 都是全局的
def skinning(joint_translation, joint_orientation, T_pose_joint_translation, T_pose_vertex_translation, skinning_idx, skinning_weight):
    """
    skinning函数，给出一桢骨骼的位姿，计算蒙皮顶点的位置
    假设M个关节，N个蒙皮顶点，每个顶点受到最多4个关节影响
    输入：
        joint_translation: (M,3)的ndarray, 目标关节的位置
        joint_orientation: (M,4)的ndarray, 目标关节的旋转，用四元数表示
        T_pose_joint_translation: (M,3)的ndarray, T pose下关节的位置
        T_pose_vertex_translation: (N,3)的ndarray, T pose下蒙皮顶点的位置
        skinning_idx: (N,4)的ndarray, 每个顶点受到哪些关节的影响（假设最多受4个关节影响）
        skinning_weight: (N,4)的ndarray, 每个顶点受到对应关节影响的权重
    输出：
        vertex_translation: (N,3)的ndarray, 蒙皮顶点的位置
    """
    vertex_translation = T_pose_vertex_translation.copy()
    
    #---------------你的代码------------------#
    # 根据T_pose下相应节点的位置和旋转,一共有16340个顶点, Linear Blend Skinning蒙皮绑定计算

    # 使用for循环，不进行numpy矩阵运算优化，3.3fps
    # for i in range(len(vertex_translation)):
    #     tmp_data = np.array([0.0, 0.0, 0.0])
    #     for j in range(4):
    #         joint_idx = skinning_idx[i][j]
    #         wij = skinning_weight[i][j]
    #         rij = R.from_quat(joint_orientation[joint_idx]).as_matrix() @ (vertex_translation[i] - T_pose_joint_translation[joint_idx])
    #         tmp_data += wij * ( rij + joint_translation[joint_idx])
    #     vertex_translation[i] = tmp_data

    # 矩阵运算优化
    N = len(skinning_idx)
    rotation = np.reshape(R.from_quat(np.reshape(joint_orientation[skinning_idx], [N * 4, 4])).as_matrix(), [N, 4, 3, 3])
    joint_target = joint_translation[skinning_idx]
    joint_origin = T_pose_joint_translation[skinning_idx]
    tmp1_tmp = np.einsum("...ijk,...i->...jk", rotation, skinning_weight)
    tmp1 = np.einsum("...jk,...k->...j", tmp1_tmp, vertex_translation)
    tmp2_tmp = np.einsum("...jk,...k->...j", rotation, joint_origin)
    tmp2 = np.einsum("...ij,...i->...j", tmp2_tmp, skinning_weight)
    tmp3 = np.einsum("...ij,...i->...j", joint_target, skinning_weight)

    vertex_translation = tmp1 - tmp2 + tmp3
    return vertex_translation