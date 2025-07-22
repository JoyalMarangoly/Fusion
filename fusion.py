import threading
import time
import csv
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from neo import get_gps_position, get_gps_status
from bno import get_linear_acceleration, get_calibration_status
from pyproj import Proj, transform
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import io
import json

# Kalman filter parameters
DT_IMU = 0.01  # IMU update rate (s)
DT_GPS = 1.0   # GPS update rate (s)
STATE_SIZE = 6 # [x, x_dot, y, y_dot, z, z_dot]

# Reference point for ENU conversion (set on first GPS fix)
ref_lat = None
ref_lon = None
ref_alt = None

# pyproj setup for WGS84 to ENU
wgs84 = Proj(proj='latlong', datum='WGS84')
enu = Proj(proj='utm', zone=33, ellps='WGS84')  # zone will be set dynamically

# Logging
LOG_HEADER = ['timestamp','x_pos','y_pos','z_pos','x_vel','y_vel','z_vel',
              'gps_x','gps_y','gps_z','acc_x','acc_y','acc_z', 'latitude', 'longitude']
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Shared state
log_lock = threading.Lock()
recording = False
kalman_thread_obj = None
current_log_file = None

uploaded_data = None  # For storing uploaded CSV data for plotting

# --- Kalman filter process noise helper ---
def make_Q(dt, sigma_a=1.0):
    """
    Generates the process noise covariance matrix Q for a 3D constant acceleration model.
    sigma_a: expected acceleration noise (m/s^2)
    """
    dt2 = dt * dt
    dt3 = dt * dt2 / 2
    dt4 = dt * dt3 / 2
    q_1d = np.array([[dt4, dt3], [dt3, dt2]]) * (sigma_a**2)
    Q = np.kron(np.eye(3), q_1d)
    return Q

# Kalman filter class
class Kalman3D:
    def __init__(self):
        self.x = np.zeros((STATE_SIZE, 1))  # [x, x_dot, y, y_dot, z, z_dot]
        self.P = np.eye(STATE_SIZE) * 10
        self.F = np.eye(STATE_SIZE)
        self.B = np.zeros((STATE_SIZE, 3))
        self.Q = np.eye(STATE_SIZE) * 0.01
        self.R = np.eye(3) * 0.01  # Decreased measurement noise
        self.H = np.zeros((3, STATE_SIZE))
        self.H[0,0] = 1
        self.H[1,2] = 1
        self.H[2,4] = 1

    def predict(self, acc, dt):
        self.F[0,1] = dt
        self.F[2,3] = dt
        self.F[4,5] = dt
        self.B[0,0] = 0.5 * dt**2
        self.B[1,0] = dt
        self.B[2,1] = 0.5 * dt**2
        self.B[3,1] = dt
        self.B[4,2] = 0.5 * dt**2
        self.B[5,2] = dt
        # Use time-scaled process noise
        self.Q = make_Q(dt, sigma_a=1.0)  # Tune sigma_a as needed
        u = np.array(acc).reshape((3,1))
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, gps):
        z = np.array(gps).reshape((3,1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(STATE_SIZE) - np.dot(K, self.H)), self.P)

# Utility: get new log file path
def new_log_file():
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(LOG_DIR, f'log_{ts}.csv')

def gps_to_meters(lat, lon, alt):
    global ref_lat, ref_lon, ref_alt, enu
    if ref_lat is None:
        ref_lat, ref_lon, ref_alt = lat, lon, alt
        utm_zone = int((ref_lon + 180) / 6) + 1
        enu = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    x0, y0 = enu(ref_lon, ref_lat)
    x, y = enu(lon, lat)
    return x - x0, y - y0, alt - ref_alt

def kalman_thread_func(log_file):
    kf = Kalman3D()
    last_gps = time.time()
    gps_x, gps_y, gps_z = 0, 0, 0
    lat, lon = None, None
    started_logging = False
    with log_lock:
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_HEADER)
    while recording:
        t0 = time.time()
        acc = get_linear_acceleration() or (0,0,0)
        # IMU outlier rejection
        if np.linalg.norm(acc) > 2.0:
            acc = (0,0,0)
        kf.predict(acc, DT_IMU)
        # Zero-velocity update (ZUPT) if stationary
        if np.linalg.norm(acc) < 0.05:
            kf.x[1,0] = 0
            kf.x[3,0] = 0
            kf.x[5,0] = 0
        if time.time() - last_gps >= 0.01:
            gps_data = get_gps_position()
            if gps_data:
                lat, lon, alt = gps_data
                gps_x, gps_y, gps_z = gps_to_meters(lat, lon, alt)
                kf.update([gps_x, gps_y, gps_z])
            last_gps = time.time()
        # Only start logging after first non-zero GPS position
        if not started_logging:
            if any([abs(gps_x) > 1e-6, abs(gps_y) > 1e-6, abs(gps_z) > 1e-6]):
                started_logging = True
            else:
                time.sleep(max(0, DT_IMU - (time.time() - t0)))
                continue
        with log_lock:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    kf.x[0,0], kf.x[2,0], kf.x[4,0],
                    kf.x[1,0], kf.x[3,0], kf.x[5,0],
                    gps_x, gps_y, gps_z,
                    acc[0], acc[1], acc[2],
                    lat, lon
                ])
        time.sleep(max(0, DT_IMU - (time.time() - t0)))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    return render_template('visualize.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, kalman_thread_obj, current_log_file
    if not recording:
        recording = True
        current_log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        kalman_thread_obj = threading.Thread(target=kalman_thread_func, args=(current_log_file,), daemon=True)
        kalman_thread_obj.start()
        return jsonify({'status': 'started', 'log_file': os.path.basename(current_log_file)})
    return jsonify({'status': 'already_running'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return jsonify({'status': 'stopped'})

@app.route('/log_data')
def log_data():
    if not current_log_file or not os.path.exists(current_log_file):
        return jsonify([])
    with log_lock:
        with open(current_log_file, 'r') as f:
            return jsonify(list(csv.DictReader(f)))  # Return all data

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    return send_from_directory(LOG_DIR, filename, as_attachment=True, mimetype='text/csv')

@app.route('/list_logs')
def list_logs():
    logs = sorted([f for f in os.listdir(LOG_DIR) if f.startswith('log_') and f.endswith('.csv')], reverse=True)
    return jsonify(logs)

@app.route('/get_log_data/<filename>')
def get_log_data(filename):
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    with open(filepath, 'r') as f:
        rows = list(csv.DictReader(f))
    x = [float(r['x_pos']) for r in rows if r.get('x_pos')]
    y = [float(r['y_pos']) for r in rows if r.get('y_pos')]
    z = [float(r['z_pos']) for r in rows if r.get('z_pos')]
    return jsonify({'x': x, 'y': y, 'z': z})

@app.route('/calibrate')
def calibrate():
    return render_template('calibrate.html')

@app.route('/calibration_status')
def calibration_status():
    # IMU calibration
    try:
        sys, gyro, accel, mag = get_calibration_status()
    except Exception:
        sys, gyro, accel, mag = None, None, None, None
    # GPS status
    try:
        fix_type, num_sats = get_gps_status()
    except Exception:
        fix_type, num_sats = None, None
    return jsonify({
        'imu': {'sys': sys, 'gyro': gyro, 'accel': accel, 'mag': mag},
        'gps': {'fix_type': fix_type, 'num_sats': num_sats}
    })

# --- 2D Kalman filter process noise helper ---
def make_Q_2d(dt, sigma_a=1.0):
    dt2 = dt * dt
    dt3 = dt * dt2 / 2
    dt4 = dt * dt3 / 2
    q_1d = np.array([[dt4, dt3], [dt3, dt2]]) * (sigma_a**2)
    Q = np.kron(np.eye(2), q_1d)
    return Q

class Kalman2D:
    def __init__(self):
        self.x = np.zeros((4, 1))  # [x, x_dot, y, y_dot]
        self.P = np.eye(4) * 10
        self.F = np.eye(4)
        self.B = np.zeros((4, 2))
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.01  # Decreased measurement noise
        self.H = np.zeros((2, 4))
        self.H[0,0] = 1
        self.H[1,2] = 1

    def predict(self, acc, dt):
        self.F[0,1] = dt
        self.F[2,3] = dt
        self.B[0,0] = 0.5 * dt**2
        self.B[1,0] = dt
        self.B[2,1] = 0.5 * dt**2
        self.B[3,1] = dt
        self.Q = make_Q_2d(dt, sigma_a=1.0)
        u = np.array(acc[:2]).reshape((2,1))
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, gps):
        z = np.array(gps[:2]).reshape((2,1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

# 2D log header
LOG_HEADER_2D = ['timestamp','x_pos','y_pos','x_vel','y_vel','gps_x','gps_y','acc_x','acc_y','latitude','longitude','filter_gps_x_err','filter_gps_y_err']

# Shared state for 2D
log_lock_2d = threading.Lock()
recording_2d = False
kalman_thread_obj_2d = None
current_log_file_2d = None

def kalman_thread_func_2d(log_file):
    kf = Kalman2D()
    last_gps = time.time()
    gps_x, gps_y = 0, 0
    lat, lon = None, None
    started_logging = False
    with log_lock_2d:
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                csv.writer(f).writerow(LOG_HEADER_2D)
    while recording_2d:
        t0 = time.time()
        acc = get_linear_acceleration() or (0,0,0)
        # IMU outlier rejection
        if np.linalg.norm(acc[:2]) > 2.0:
            acc = (0,0,acc[2])
        kf.predict(acc, DT_IMU)
        # Zero-velocity update (ZUPT) if stationary
        if np.linalg.norm(acc[:2]) < 0.05:
            kf.x[1,0] = 0
            kf.x[3,0] = 0
        if time.time() - last_gps >= 0.01:
            gps_data = get_gps_position()
            if gps_data:
                lat, lon, alt = gps_data
                gps_x, gps_y, _ = gps_to_meters(lat, lon, 0)
                kf.update([gps_x, gps_y])
            last_gps = time.time()
        # Only start logging after first non-zero GPS position
        if not started_logging:
            if any([abs(gps_x) > 1e-6, abs(gps_y) > 1e-6]):
                started_logging = True
            else:
                time.sleep(max(0, DT_IMU - (time.time() - t0)))
                continue
        filter_x = kf.x[0,0]
        filter_y = kf.x[2,0]
        err_x = filter_x - gps_x
        err_y = filter_y - gps_y
        with log_lock_2d:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.utcnow().isoformat(),
                    filter_x, filter_y,
                    kf.x[1,0], kf.x[3,0],
                    gps_x, gps_y,
                    acc[0], acc[1],
                    lat, lon,
                    err_x, err_y
                ])
        time.sleep(max(0, DT_IMU - (time.time() - t0)))

@app.route('/start_recording_2d', methods=['POST'])
def start_recording_2d():
    global recording_2d, kalman_thread_obj_2d, current_log_file_2d
    if not recording_2d:
        recording_2d = True
        current_log_file_2d = os.path.join(LOG_DIR, f"log2d_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        kalman_thread_obj_2d = threading.Thread(target=kalman_thread_func_2d, args=(current_log_file_2d,), daemon=True)
        kalman_thread_obj_2d.start()
        return jsonify({'status': 'started', 'log_file': os.path.basename(current_log_file_2d)})
    return jsonify({'status': 'already_running'})

@app.route('/stop_recording_2d', methods=['POST'])
def stop_recording_2d():
    global recording_2d
    recording_2d = False
    return jsonify({'status': 'stopped'})

@app.route('/list_logs_2d')
def list_logs_2d():
    logs = sorted([f for f in os.listdir(LOG_DIR) if f.startswith('log2d_') and f.endswith('.csv')], reverse=True)
    return jsonify(logs)

@app.route('/get_log_data_2d/<filename>')
def get_log_data_2d(filename):
    filepath = os.path.join(LOG_DIR, filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    with open(filepath, 'r') as f:
        rows = list(csv.DictReader(f))
    x = [float(r['x_pos']) for r in rows if r.get('x_pos')]
    y = [float(r['y_pos']) for r in rows if r.get('y_pos')]
    gps_x = [float(r['gps_x']) for r in rows if r.get('gps_x')]
    gps_y = [float(r['gps_y']) for r in rows if r.get('gps_y')]
    err_x = [float(r['filter_gps_x_err']) for r in rows if r.get('filter_gps_x_err')]
    err_y = [float(r['filter_gps_y_err']) for r in rows if r.get('filter_gps_y_err')]
    return jsonify({'x': x, 'y': y, 'gps_x': gps_x, 'gps_y': gps_y, 'err_x': err_x, 'err_y': err_y})

@app.route('/visualize2d')
def visualize2d():
    return render_template('visualize2d.html')

@app.route('/ntrip_status')
def ntrip_status():
    status_file = os.path.join(os.path.dirname(__file__), 'ntrip_status.json')
    if not os.path.exists(status_file):
        return jsonify({'status': 'unknown'})
    with open(status_file, 'r') as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False) 