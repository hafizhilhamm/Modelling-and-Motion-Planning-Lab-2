from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray
import math

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    if T.shape != (3, 3):
        raise ValueError
    
    q = Quaternion()
    trace = np.trace(T)
    if trace > 0:
        # w is the largest component
        s = 2.0 * np.sqrt(trace + 1.0)
        q.w = s /4.0
        q.x = (T[2,1] - T[1,2]) / s
        q.y = (T[0,2] - T[2,0]) / s
        q.z = (T[1,0] - T[0,1]) / s

    elif T[0,0] > T[1,1] and T[0,0] > T[2,2]:
        # x is the largest component
        s = 2.0 * np.sqrt(
        1.0 + T[0,0] - T[1,1] - T[2,2]
        )

        q.x = s / 4.0
        q.w = (T[2,1] - T[1,2]) / s
        q.y = (T[0,1] + T[1,0]) / s
        q.z = (T[0,2] + T[2,0]) / s
    elif T[1,1] > T[2,2]:
        # y is the largest component
        s = 2.0 * np.sqrt(
            1.0 + T[1,1] - T[0,0] - T[2,2]
        )

        q.y = s / 4.0
        q.w = (T[0,2] - T[2,0]) / s
        q.x = (T[0,1] + T[1,0]) / s
        q.z = (T[1,2] + T[2,1]) / s

    else:
        # z is the largest component
        s = 2.0 * np.sqrt(
            1.0 + T[2,2] - T[0,0] - T[1,1]
        )

        q.z = s / 4.0
        q.w = (T[1,0] - T[0,1]) / s
        q.x = (T[0,2] + T[2,0]) / s
        q.y = (T[1,2] + T[2,1]) / s
    return q
def quat2rodrigues(q: Quaternion) -> NDArray:
    w,x,y,z = q.w , q.x, q.y, q.z #quaternion x,y,z, and w

    # q and -q are the same rotation, flip so the angle stays small
    if w < 0 :
        x,y,z,w = x * - 1, y * - 1, z* - 1, w * - 1
    
    # length of (x,y,z) = sin(half angle)
    s = math.sqrt(x ** 2 + y ** 2 + z ** 2 )

    # no rotation(s is almost 0), axis is undefined, so return zero vector
    if s < 1e-9:
        T = np.array([0,0,0])
        return T

    theta = np.arctan2(s, w) * 2
    axis = np.array([x, y, z]) / s
    T = theta * axis
    return T

def quat2zyx(q: Quaternion) -> NDArray:
    w,x,y,z = q.w , q.x, q.y, q.z

    # 5 numbers from the rotation matrix, calculated from the quaternion
    r00 = 1 - 2*(y**2 + z**2) 
    r10 = 2 * (x*y + w*z)
    r20 = 2 * (x*z - w*y)
    r21 = 2 * (y*z + w*x)
    r22 = 1 - 2 * (x**2 + y**2)

    #count yaw, pitch, and roll
    yaw = np.arctan2(r10, r00)
    roll = np.arctan2(r21,r22)
    pitch = np.arcsin((np.clip(r20 * -1, -1, 1)))

    T = np.array([yaw, pitch, roll])

    return T