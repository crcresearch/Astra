import random
from stream import BaseStream

class SensorArrayStream(BaseStream):
    def start(self):
        print("[SENSOR] SensorArrayStream start method not implemented")

    def stop(self):
        print("[SENSOR] SensorArrayStream stop method not implemented")

    def read(self):
        print("[SENSOR] SensorArrayStream read method not implemented")

    def simulate_read(self):
        simulated_values = {
            "temperature": round(random.uniform(20, 35), 2),
            "humidity": random.randint(20, 80),
            "light": random.randint(0, 100),
            "gas": round(random.uniform(0, 1), 4),
        }
        return simulated_values