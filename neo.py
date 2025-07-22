import time
import serial
import pynmea2

port = serial.Serial('/dev/ttyACM0', baudrate = 38400, timeout = 1)

def get_gps_position():
    starttime = time.time()
    while(time.time() - starttime <0.9):
        line = port.readline().decode('ascii', errors = 'replace')
        if line.startswith('$GPGGA') or line.startswith('$GNGGA'):  
            msg = pynmea2.parse(line)
            lat = float(msg.latitude)
            lon = float(msg.longitude)
            alt = float(msg.altitude)
            return lat,lon,alt
        
def get_gps_status():
    starttime = time.time()
    while(time.time() - starttime <0.9):
        line = port.readline().decode('ascii', errors = 'replace')
        if line.startswith('$GPGGA') or line.startswith('$GNGGA'):  
            msg = pynmea2.parse(line)
            num_sats = int(msg.num_sats)
            quality = msg.gps_qual
            return quality, num_sats

if __name__ == '__main__':
    while True:
        line = port.readline().decode('ascii', errors='replace')
        if line.startswith('$GPGGA') or line.startswith('$GNGGA'):
            msg = pynmea2.parse(line)
            print(f"Latitude: {msg.latitude}")
            print(f"Longitude: {msg.longitude}")
            print(f"Altitude: {msg.altitude} {msg.altitude_units}")
            print(f"Satellites locked: {msg.num_sats}")
            print(f"Fix Quality: {msg.gps_qual}")