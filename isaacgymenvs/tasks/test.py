# import torch
# from scipy.spatial.transform import Rotation

# def Euler_ang_from_quat(quaternions):
#         """
#         Convert a tensor of quaternions to Euler angles (roll, pitch, yaw).
        
#         Args:
#             quaternions (torch.Tensor): A tensor of shape (..., 4), where the last dimension 
#                                         represents quaternion components (x, y, z, w).
                                        
#         Returns:
#             torch.Tensor: A tensor of shape (..., 3), containing the Euler angles 
#                         (roll, pitch, yaw) corresponding to the input quaternions.
#         """
#         # Ensure quaternions are normalized
#         quaternions = quaternions / torch.norm(quaternions, dim=-1, keepdim=True)
        
#         # Extract components of the quaternion
#         x, y, z, w = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
        
#         # Compute the Euler angles
#         t0 = +2.0 * (w * x + y * z)
#         t1 = +1.0 - 2.0 * (x * x + y * y)
#         X = torch.atan2(t0, t1)
     
#         t2 = +2.0 * (w * y - z * x)
#         t2 = +1.0 if t2 > +1.0 else t2
#         t2 = -1.0 if t2 < -1.0 else t2
#         Y = torch.asin(t2)
     
#         t3 = +2.0 * (w * z + x * y)
#         t4 = +1.0 - 2.0 * (y * y + z * z)
#         Z = torch.atan2(t3, t4)
        
#         return torch.stack([X, Y, Z], dim=-1)

# quaternions = torch.tensor([[0.1262852, 0.1261165, 0.2100786, 0.9612563]], dtype=torch.float32)
# print(Euler_ang_from_quat(quaternions))

import math
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + z * z)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

print(euler_from_quaternion(0.1262852, 0.1261165, 0.2100786, 0.9612563))