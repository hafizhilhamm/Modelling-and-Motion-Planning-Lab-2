from geometry_msgs.msg import Quaternion

import numpy as np
from numpy.typing import NDArray

def rotmat2q(T: NDArray) -> Quaternion:
    # Function that transforms a 3x3 rotation matrix to a ros quaternion representation
    if T.shape != (3, 3):
        raise ValueError
    
    q = Quaternion()
    trace = np.trace(T)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q.w = s /4.0
        q.x = (T[2,1] - T[1,2]) / s
        q.y = (T[0,2] - T[2,0]) / s
        q.z = (T[1,0] - T[0,1]) / s

    elif T[0,0] > T[1,1] and T[0,0] > T[2,2]:
        s = 2.0 * np.sqrt(
        1.0 + T[0,0] - T[1,1] - T[2,2]
        )

        q.x = s / 4.0
        q.w = (T[2,1] - T[1,2]) / s
        q.y = (T[0,1] + T[1,0]) / s
        q.z = (T[0,2] + T[2,0]) / s
    elif T[1,1] > T[2,2]:
        s = 2.0 * np.sqrt(
            1.0 + T[1,1] - T[0,0] - T[2,2]
        )

        q.y = s / 4.0
        q.w = (T[0,2] - T[2,0]) / s
        q.x = (T[0,1] + T[1,0]) / s
        q.z = (T[1,2] + T[2,1]) / s

    else:
        s = 2.0 * np.sqrt(
            1.0 + T[2,2] - T[0,0] - T[1,1]
        )

        q.z = s / 4.0
        q.w = (T[1,0] - T[0,1]) / s
        q.x = (T[0,2] + T[2,0]) / s
        q.y = (T[1,2] + T[2,1]) / s
    return q
