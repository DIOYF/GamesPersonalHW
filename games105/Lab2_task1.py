import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
# ------------- lab1里的代码 -------------#
def load_meta_data(bvh_path):
    with open(bvh_path, 'r') as f:
        channels = []
        joints = []
        joint_parents = []
        joint_offsets = []
        end_sites = []

        parent_stack = [None]
        for line in f:
            if 'ROOT' in line or 'JOINT' in line:
                joints.append(line.split()[-1])
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif 'End Site' in line:
                end_sites.append(len(joints))
                joints.append(parent_stack[-1] + '_end')
                joint_parents.append(parent_stack[-1])
                channels.append('')
                joint_offsets.append([0, 0, 0])

            elif '{' in line:
                parent_stack.append(joints[-1])

            elif '}' in line:
                parent_stack.pop()

            elif 'OFFSET' in line:
                joint_offsets[-1] = np.array([float(x) for x in line.split()[-3:]]).reshape(1,3)

            elif 'CHANNELS' in line:
                trans_order = []
                rot_order = []
                for token in line.split():
                    if 'position' in token:
                        trans_order.append(token[0])

                    if 'rotation' in token:
                        rot_order.append(token[0])

                channels[-1] = ''.join(trans_order)+ ''.join(rot_order)

            elif 'Frame Time:' in line:
                break
        
    joint_parents = [-1]+ [joints.index(i) for i in joint_parents[1:]]
    channels = [len(i) for i in channels]
    return joints, joint_parents, channels, joint_offsets

def load_motion_data(bvh_path):
    with open(bvh_path, 'r') as f:
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

# ------------- 实现一个简易的BVH对象，进行数据处理 -------------#

'''
注释里统一N表示帧数，M表示关节数
position, rotation表示局部平移和旋转
translation, orientation表示全局平移和旋转
'''

class BVHMotion():
    def __init__(self, bvh_file_name = None) -> None:
        
        # 一些 meta data
        self.joint_name = []
        self.joint_channel = []
        self.joint_parent = []
        
        # 一些local数据, 对应bvh里的channel, XYZposition和 XYZrotation
        #! 这里我们把没有XYZ position的joint的position设置为offset, 从而进行统一
        self.joint_position = None # (N,M,3) 的ndarray, 局部平移
        self.joint_rotation = None # (N,M,4)的ndarray, 用四元数表示的局部旋转
        
        if bvh_file_name is not None:
            self.load_motion(bvh_file_name)
        pass
    
    #------------------- 一些辅助函数 ------------------- #
    def load_motion(self, bvh_file_path):
        '''
            读取bvh文件，初始化元数据和局部数据
        '''
        self.joint_name, self.joint_parent, self.joint_channel, joint_offset = \
            load_meta_data(bvh_file_path)
        
        motion_data = load_motion_data(bvh_file_path)

        # 把motion_data里的数据分配到joint_position和joint_rotation里
        self.joint_position = np.zeros((motion_data.shape[0], len(self.joint_name), 3))
        self.joint_rotation = np.zeros((motion_data.shape[0], len(self.joint_name), 4))
        self.joint_rotation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        cur_channel = 0
        for i in range(len(self.joint_name)):
            if self.joint_channel[i] == 0:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                continue   
            elif self.joint_channel[i] == 3:
                self.joint_position[:,i,:] = joint_offset[i].reshape(1,3)
                rotation = motion_data[:, cur_channel:cur_channel+3]
            elif self.joint_channel[i] == 6:
                self.joint_position[:, i, :] = motion_data[:, cur_channel:cur_channel+3]
                rotation = motion_data[:, cur_channel+3:cur_channel+6]
            self.joint_rotation[:, i, :] = R.from_euler('XYZ', rotation,degrees=True).as_quat()
            cur_channel += self.joint_channel[i]
        
        return

    def batch_forward_kinematics(self, joint_position = None, joint_rotation = None):
        '''
        利用自身的metadata进行批量前向运动学
        joint_position: (N,M,3)的ndarray, 局部平移
        joint_rotation: (N,M,4)的ndarray, 用四元数表示的局部旋转
        '''
        if joint_position is None:
            joint_position = self.joint_position
        if joint_rotation is None:
            joint_rotation = self.joint_rotation
        
        joint_translation = np.zeros_like(joint_position)
        joint_orientation = np.zeros_like(joint_rotation)
        joint_orientation[:,:,3] = 1.0 # 四元数的w分量默认为1
        
        # 一个小hack是root joint的parent是-1, 对应最后一个关节
        # 计算根节点时最后一个关节还未被计算，刚好是0偏移和单位朝向
        
        for i in range(len(self.joint_name)):
            pi = self.joint_parent[i]
            parent_orientation = R.from_quat(joint_orientation[:,pi,:]) 
            joint_translation[:, i, :] = joint_translation[:, pi, :] + \
                parent_orientation.apply(joint_position[:, i, :])
            joint_orientation[:, i, :] = (parent_orientation * R.from_quat(joint_rotation[:, i, :])).as_quat()
        return joint_translation, joint_orientation
    
    
    def adjust_joint_name(self, target_joint_name):
        '''
        调整关节顺序为target_joint_name
        '''
        idx = [self.joint_name.index(joint_name) for joint_name in target_joint_name]
        idx_inv = [target_joint_name.index(joint_name) for joint_name in self.joint_name]
        self.joint_name = [self.joint_name[i] for i in idx]
        self.joint_parent = [idx_inv[self.joint_parent[i]] for i in idx]
        self.joint_parent[0] = -1
        self.joint_channel = [self.joint_channel[i] for i in idx]
        self.joint_position = self.joint_position[:,idx,:]
        self.joint_rotation = self.joint_rotation[:,idx,:]
        pass
    
    def raw_copy(self):
        '''
        返回一个拷贝
        '''
        return copy.deepcopy(self)
    
    @property
    def motion_length(self):
        return self.joint_position.shape[0]
    
    
    def sub_sequence(self, start, end):
        '''
        返回一个子序列
        start: 开始帧
        end: 结束帧
        '''
        res = self.raw_copy()
        res.joint_position = res.joint_position[start:end, :, :]
        res.joint_rotation = res.joint_rotation[start:end, :, :]
        return res
    
    def append(self, other):
        '''
        在末尾添加另一个动作
        '''
        other = other.raw_copy()
        other.adjust_joint_name(self.joint_name)
        self.joint_position = np.concatenate((self.joint_position, other.joint_position), axis=0)
        self.joint_rotation = np.concatenate((self.joint_rotation, other.joint_rotation), axis=0)
        pass



    #--------------------- 你的任务 -------------------- #
    
    def decompose_rotation_with_yaxis(self, rotation):
        '''
        输入: rotation 形状为(4,)的ndarray, 四元数旋转
        输出: Ry, Rxz，分别为绕y轴的旋转和转轴在xz平面的旋转，并满足R = Ry * Rxz
        '''
        # TODO: 你的代码
        # 将四元数旋转分解为绕y轴的旋转，和转轴在xz平面的旋转，先得到Ry，再逆运算得到Rxz
        Ry = R.from_quat(rotation).as_euler("XYZ", degrees=True)
        Ry = R.from_euler("XYZ", [0, Ry[1], 0], degrees=True)

        # Ry = R.from_quat(rotation).

        Rxz = Ry.inv() * R.from_quat(rotation)
        return Ry, Rxz
    
    # part 1
    def translation_and_rotation(self, frame_num, target_translation_xz, target_facing_direction_xz):
        '''
        计算出新的joint_position和joint_rotation
        使第frame_num帧的根节点平移为target_translation_xz, 水平面朝向为target_facing_direction_xz
        frame_num: int
        target_translation_xz: (2,)的ndarray
        target_faceing_direction_xz: (2,)的ndarray，表示水平朝向。你可以理解为原本的z轴被旋转到这个方向。
        Tips:
            主要是调整root节点的joint_position和joint_rotation
            frame_num可能是负数，遵循python的索引规则
            你需要完成并使用decompose_rotation_with_yaxis
            输入的target_facing_direction_xz的norm不一定是1
        '''
        
        res = self.raw_copy() # 拷贝一份，不要修改原始数据
        
        # 比如说，你可以这样调整第frame_num帧的根节点平移
        offset = target_translation_xz - res.joint_position[frame_num, 0, [0,2]]
        res.joint_position[:, 0, [0, 2]] += offset
        # TODO: 你的代码

        sin_theta_xz = np.cross(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
        cos_theta_xz = np.dot(target_facing_direction_xz, np.array([0, 1])) / np.linalg.norm(target_facing_direction_xz)
        theta = np.arccos(cos_theta_xz)
        if sin_theta_xz < 0:
            theta = 2 * np.pi - theta
        new_root_Ry = R.from_euler("Y", theta, degrees=False)
        R_y, _ = self.decompose_rotation_with_yaxis(res.joint_rotation[frame_num, 0, :])

        res.joint_rotation[:, 0, :] = (new_root_Ry * R_y.inv() * R.from_quat(res.joint_rotation[:, 0, :])).as_quat()
        for i in range(len(res.joint_position)):
             res.joint_position[i, 0,:] = (new_root_Ry * R_y.inv()).as_matrix()  @ (res.joint_position[i, 0, :] - res.joint_position[frame_num, 0, :]) + res.joint_position[frame_num,0,:]

        return res


def slerp(joint_rotation_1, joint_rotation_2, alpha):
    rotation_a = joint_rotation_1
    rotation_b = joint_rotation_2
    cos_half_theta = np.dot(rotation_a, rotation_b)

    if cos_half_theta < 0.:
        cos_half_theta = -cos_half_theta
        rotation_a = -rotation_a

    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sin(half_theta)

    if sin_half_theta > 0.001:
        alpha_1 = np.sin((1 - alpha) * half_theta) / sin_half_theta
        alpha_2 = np.sin(alpha* half_theta) / sin_half_theta
    else:
        alpha_1 = 1 - alpha
        alpha_2 = alpha
    res_joint_rotation = alpha_1 * rotation_a + alpha_2 * rotation_b
    res_joint_rotation /= np.linalg.norm(res_joint_rotation)
    return res_joint_rotation


# part2
def blend_two_motions(bvh_motion1, bvh_motion2, alpha):
    '''
    blend两个bvh动作
    假设两个动作的帧数分别为n1, n2
    alpha: 0~1之间的浮点数组，形状为(n3,)
    返回的动作应该有n3帧，第i帧由(1-alpha[i]) * bvh_motion1[j] + alpha[i] * bvh_motion2[k]得到
    i均匀地遍历0~n3-1的同时，j和k应该均匀地遍历0~n1-1和0~n2-1
    '''
    
    res = bvh_motion1.raw_copy()
    res.joint_position = np.zeros((len(alpha), res.joint_position.shape[1], res.joint_position.shape[2]))
    res.joint_rotation = np.zeros((len(alpha), res.joint_rotation.shape[1], res.joint_rotation.shape[2]))
    res.joint_rotation[..., 3] = 1.0

    # TODO: 你的代码
    n_1 = len(bvh_motion1.joint_position)
    n_2 = len(bvh_motion2.joint_position)
    n_3 = len(alpha)
    # linear interporation
    for i in range(0, n_3):
        j =int( (i * n_1) /n_3)
        k =int( (i * n_2) /n_3)
        res.joint_position[i, :, :] = (1-alpha[i]) * bvh_motion1.joint_position[j, :, :] + (alpha[i]) * bvh_motion2.joint_position[k, :, :]

        # 四元数s_leap:
        for l in range(0, len(res.joint_rotation[0])):
            res.joint_rotation[i, l, :] = slerp(bvh_motion1.joint_rotation[j,l,:], bvh_motion2.joint_rotation[k, l, :], alpha[i])

            # cos_half_theta = np.dot( bvh_motion1.joint_rotation[j,l,:], bvh_motion2.joint_rotation[k, l, :])
            # temp_b = bvh_motion1.joint_rotation[j,l,:]
            #
            # if cos_half_theta < 0.:
            #     cos_half_theta = -cos_half_theta
            #     temp_b = -bvh_motion1.joint_rotation[j, l,:]
            #
            # half_theta = np.arccos(cos_half_theta)
            # sin_half_theta = np.sin(half_theta)
            #
            # if sin_half_theta > 0.001:
            #     alpha_1 = np.sin( (1-alpha[i]) * half_theta) / sin_half_theta
            #     alpha_2 = np.sin( alpha[i] * half_theta) / sin_half_theta
            # else:
            #     alpha_1 = 1- alpha[i]
            #     alpha_2 = alpha[i]
            # res.joint_rotation[i,l,:] = alpha_1 * temp_b + alpha_2 * bvh_motion2.joint_rotation[k, l,:]
            # res.joint_rotation[i,l,:] /= np.linalg.norm(res.joint_rotation[i,l,:])

    return res

# part3
def build_loop_motion(bvh_motion):
    '''
    将bvh动作变为循环动作
    由于比较复杂,作为福利,不用自己实现
    (当然你也可以自己实现试一下)
    推荐阅读 https://theorangeduck.com/
    Creating Looping Animations from Motion Capture
    '''
    res = bvh_motion.raw_copy()
    
    from smooth_utils import build_loop_motion
    return build_loop_motion(res)

# part4
def concatenate_two_motions(bvh_motion1, bvh_motion2, mix_frame1, mix_time):
    '''
    将两个bvh动作平滑地连接起来，mix_time表示用于混合的帧数
    混合开始时间是第一个动作的第mix_frame1帧,
    虽然某些混合方法可能不需要mix_time，但是为了保证接口一致，我们还是保留这个参数
    Tips:
        你可能需要用到BVHMotion.sub_sequence 和 BVHMotion.append
    '''
    res = bvh_motion1.raw_copy()
    
    # TODO: 你的代码
    # 下面这种直接拼肯定是不行的(
    # 从mix_frame1截断， 先播放frame1，再播放frame2，这种肯定是不对的，最起码对动作进行一个转换，对吧

    # 从mix_frame开始的动作到新动作的第一帧对齐
    rot = bvh_motion1.joint_rotation[mix_frame1, 0]
    facing_axis = R.from_quat(rot).apply(np.array([0, 0, 1])).flatten()[[0, 2]]

    new_bvh_motion2 = bvh_motion2.translation_and_rotation(0, bvh_motion1.joint_position[mix_frame1, 0, [0, 2]], facing_axis)

    # 进行动画blending,分别使用惯性混合和线性插值方法
    blending_joint_position = np.zeros((mix_time, new_bvh_motion2.joint_position.shape[1], new_bvh_motion2.joint_position.shape[2]))
    blending_joint_rotation = np.zeros((mix_time, new_bvh_motion2.joint_rotation.shape[1], new_bvh_motion2.joint_rotation.shape[2]))
    blending_joint_rotation[..., 3] = 1.0

    # 惯性方法： inertialize
    half_time = 0.3
    dt = 1 / 60
    y = 4.0 * 0.69314 / (half_time + 1e-5)

    from smooth_utils import quat_to_avel
    src_avel = quat_to_avel(bvh_motion1.joint_rotation[mix_frame1-15:mix_frame1], dt)
    dst_avel = quat_to_avel(new_bvh_motion2.joint_rotation[0:15], dt)
    off_avel = src_avel[-1] - dst_avel[0]
    off_rot = (R.from_quat(bvh_motion1.joint_rotation[mix_frame1])*R.from_quat(new_bvh_motion2.joint_rotation[0].copy()).inv()).as_rotvec()

    src_vel = bvh_motion1.joint_position[mix_frame1] - bvh_motion1.joint_position[mix_frame1-1]
    dst_vel = new_bvh_motion2.joint_position[1]-new_bvh_motion2.joint_position[0]
    off_vel = (src_vel - dst_vel) / 60
    off_pos = bvh_motion1.joint_position[mix_frame1] - new_bvh_motion2.joint_position[0]

    for i in range(len(new_bvh_motion2.joint_position)):
        tmp_ydt = y * i * dt
        eydt = np.exp( -tmp_ydt)
        # eydt = 1.0 / (1.0 + tmp_ydt + 0.48 * tmp_ydt * tmp_ydt + 0.235 * tmp_ydt * tmp_ydt * tmp_ydt)
        j1 = off_vel + off_pos * y
        j2 = off_avel + off_rot * y
        off_pos_i = eydt * (off_pos + j1 * i * dt)
        off_vel_i = eydt * (off_vel - j1 * y * i * dt)
        off_rot_i = R.from_rotvec(eydt * (off_rot + j2 * i * dt)).as_rotvec()
        off_avel_i = eydt * (off_avel - j2 * y * i * dt)


        new_bvh_motion2.joint_position[i] = new_bvh_motion2.joint_position[i] + off_pos_i
        new_bvh_motion2.joint_rotation[i] = (R.from_rotvec(off_rot_i)* R.from_quat(new_bvh_motion2.joint_rotation[i])).as_quat()


    # # 线性blending，动画增加30帧
    # for i in range(mix_time):
    #     t = i / mix_time
    #     blending_joint_position[i] = (1-t) * res.joint_position[mix_frame1] + t * new_bvh_motion2.joint_position[0]
    #     for j in range(len(res.joint_rotation[mix_frame1])):
    #         blending_joint_rotation[i, j] = slerp(res.joint_rotation[mix_frame1,j], new_bvh_motion2.joint_rotation[0,j], t)
    # new_bvh_motion2.joint_position = np.concatenate([blending_joint_position,  new_bvh_motion2.joint_position], axis=0)
    # new_bvh_motion2.joint_rotation = np.concatenate([blending_joint_rotation,  new_bvh_motion2.joint_rotation], axis=0)


    res.joint_position = np.concatenate([res.joint_position[:mix_frame1],  new_bvh_motion2.joint_position], axis=0)
    res.joint_rotation = np.concatenate([res.joint_rotation[:mix_frame1],  new_bvh_motion2.joint_rotation], axis=0)



    
    return res

