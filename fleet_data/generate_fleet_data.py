# generate a fleet of 5 vehicles with the following attributes:
# - vehicle_id
    # fixed at 5 vehicles, then each vehicle has the following attributes for each second

# basic vehicle attributes:
# - vehicle_status (moving, stopped, charging, error, accident)
# - vehicle_speed (mph)
# - timestamp (datetime)

# Advanced vehicle attributes:
# - vehicle_location (latitude, longitude)
# - vehicle_door_status (open, closed)
# - vehicle_seat_status (occupied, empty)
# - brake_lights (on/off) - indicates stopping
# - hazard_lights (on/off) - indicates emergency/disabled
# - turn_signals (left/right/off) - indicates intent
# - headlights (off/low/high) - indicates visibility conditions

# goal: print the fleet data to the console in a readable format

from datetime import datetime
import time
import random 

class Vehicle:

    def __init__(self, vehicle_id):
        available_status = ['moving', 'stopped', 'error', 'accident', 'none']
        min_speed = 0
        max_speed = 85

        self.vehicle_id = vehicle_id
        self.vehicle_status = available_status[4]
        self.vehicle_speed = random.randint(min_speed, max_speed)
        self.timestamp = datetime.now()

    def update_time(self): 
        self.timestamp = datetime.now()

    def update_speed(self):
        min_speed = 0
        max_speed = 85
        speed_change = random.randint(-10, 10)
        current_speed = self.vehicle_speed
        updated_speed = current_speed + speed_change
        if updated_speed < min_speed:
            updated_speed = min_speed
        elif updated_speed > max_speed:
            updated_speed = max_speed
        self.vehicle_speed = updated_speed

    def check_status(self):
        # To determine if the vehicle is stopped or not
        available_status = ['moving', 'stopped', 'error', 'accident', 'none']
        if self.vehicle_speed < 5:
            self.vehicle_status = available_status[1]
        else:
            self.vehicle_status = available_status[0]
    
    def __str__(self) -> str:
        return f'Vehicle: {self.vehicle_id}, Status: {self.vehicle_status} {self.vehicle_speed}mph at {self.timestamp}'

def simulate_traffic_jam(fleet, probability=0.3) -> bool:
    if random.random() < probability:
        for vehicle in fleet:
            vehicle.vehicle_speed = random.randint(0, 10)
        return True
    return False

# A fleet of 5 vehicles
fleet = [Vehicle(f'Vehicle{i}') for i in range(5)]

# get the update every second for 10 seconds
for _ in range(5):
    is_traffic = simulate_traffic_jam(fleet)

    for vehicle in fleet:
        if not is_traffic:
            vehicle.update_speed()
        vehicle.update_time()
        vehicle.check_status()
        print(vehicle)
    if is_traffic:
        print('---Traffic Jam---')
    time.sleep(1)