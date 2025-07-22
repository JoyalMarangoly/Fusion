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

def get_calibration_status():
    return sensor.calibration_status

if __name__ == '__main__':
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
        print("Calibration status: ",sensor.calibration_status)
        print("-" * 40)
        time.sleep(1)
