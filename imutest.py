import time
import board
import busio
import adafruit_bno055


# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize BNO055
sensor = adafruit_bno055.BNO055_I2C(i2c)
#syst, gyr, accel, magn = sensor.calibration_status
def get_linear_acceleration():
    return sensor.linear_acceleration

def live_3d_orientation():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib and numpy are required for visualization. Install with 'pip3 install matplotlib numpy'.")
        return

    def quat_to_rotmat(q):
        # q = (w, x, y, z)
        w, x, y, z = q
        return np.array([
            [1-2*y**2-2*z**2, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x**2-2*z**2, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x**2-2*y**2]
        ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Live IMU Orientation (BNO055)')
    plt.ion()
    plt.show()

    # Draw a reference body (axes)
    body = np.array([
        [0, 0, 0], [1, 0, 0],  # X axis (red)
        [0, 0, 0], [0, 1, 0],  # Y axis (green)
        [0, 0, 0], [0, 0, 1],  # Z axis (blue)
    ]).reshape(3, 2, 3)

    while True:
        q = sensor.quaternion
        if q is None:
            print("Waiting for quaternion data...")
            time.sleep(0.1)
            continue
        # BNO055 returns (w, x, y, z)
        rot = quat_to_rotmat(q)
        # Transform axes
        axes = np.dot(body, rot.T)
        ax.cla()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Live IMU Orientation (BNO055)')
        # X axis
        ax.plot([axes[0,0,0], axes[0,1,0]], [axes[0,0,1], axes[0,1,1]], [axes[0,0,2], axes[0,1,2]], 'r-', linewidth=3)
        # Y axis
        ax.plot([axes[1,0,0], axes[1,1,0]], [axes[1,0,1], axes[1,1,1]], [axes[1,0,2], axes[1,1,2]], 'g-', linewidth=3)
        # Z axis
        ax.plot([axes[2,0,0], axes[2,1,0]], [axes[2,0,1], axes[2,1,1]], [axes[2,0,2], axes[2,1,2]], 'b-', linewidth=3)
        plt.draw()
        plt.pause(0.01)

def live_3d_magnetometer():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from collections import deque
    except ImportError:
        print("matplotlib and numpy are required for visualization. Install with 'pip3 install matplotlib numpy'.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # To store a short history of points
    trail = deque(maxlen=100)

    plt.ion()
    plt.show()

    while True:
        mag = sensor.magnetic
        if mag is None:
            print("Waiting for magnetometer data...")
            time.sleep(0.1)
            continue

        x, y, z = mag
        trail.append((x, y, z))

        ax.cla()
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-100, 100])
        ax.set_xlabel('X (μT)')
        ax.set_ylabel('Y (μT)')
        ax.set_zlabel('Z (μT)')
        ax.set_title('Live Magnetometer Readings (BNO055)')

        # Draw magnetic vector from origin
        ax.quiver(0, 0, 0, x, y, z, color='m', length=1.0, normalize=False)

        # Draw trail
        if len(trail) > 1:
            trail_np = np.array(trail)
            ax.plot(trail_np[:, 0], trail_np[:, 1], trail_np[:, 2], 'k--', alpha=0.5)

        plt.draw()
        plt.pause(0.05)
def live_3d_accelerometer():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from collections import deque
    except ImportError:
        print("matplotlib and numpy are required for visualization. Install with 'pip3 install matplotlib numpy'.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # History of recent acceleration vectors
    trail = deque(maxlen=100)

    plt.ion()
    plt.show()

    while True:
        acc = sensor.acceleration
        if acc is None:
            print("Waiting for accelerometer data...")
            time.sleep(0.1)
            continue

        x, y, z = acc
        trail.append((x, y, z))

        ax.cla()
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])
        ax.set_xlabel('X (m/s²)')
        ax.set_ylabel('Y (m/s²)')
        ax.set_zlabel('Z (m/s²)')
        ax.set_title('Live Accelerometer Readings (BNO055)')

        # Draw the current acceleration vector
        ax.quiver(0, 0, 0, x, y, z, color='c', length=1.0, normalize=False)

        # Draw a trail of previous readings
        if len(trail) > 1:
            trail_np = np.array(trail)
            ax.plot(trail_np[:, 0], trail_np[:, 1], trail_np[:, 2], 'gray', linestyle='--', alpha=0.5)

        plt.draw()
        plt.pause(0.05)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == '--gyro':
            live_3d_orientation()
        elif sys.argv[1] == '--magn':
            live_3d_magnetometer()
        elif sys.argv[1] == '--acel':
            live_3d_accelerometer()
    else:
        # Read sensor data in a loop
        while True:
            print("Temperature:", sensor.temperature, "°C")
            print("Accelerometer (m/s^2):", sensor.acceleration)
            print("Gyroscope (rad/s):", sensor.gyro)
            print("Magnetometer (microteslas):", sensor.magnetic)
            print("Euler angle (degrees):", sensor.euler)
            print("Quaternion:", sensor.quaternion)
            print("Linear acceleration (m/s^2):", sensor.linear_acceleration)
            print("Gravity (m/s^2):", sensor.gravity)
            print("Calibration status: ", sensor.calibration_status)
            print("-" * 40)
            time.sleep(1)
