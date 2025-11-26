from confluent_kafka import Producer
import pandas as pd
import time
import json
import random
import numpy as np

# Kafka configuration
kafka_config = {'bootstrap.servers': 'localhost:9092'}

# Kafka topics
engine_topic = 'engine-sensors'
battery_topic = 'battery-sensors'

# CSV files
engine_csv = 'engines_dataset-train.csv'
battery_excel = 'charging_experiment_7_Channel_7.1 (1).xlsx'

# Initialize Kafka producers for engine and battery sensors
engine_producers = {
    "engineRpm": Producer(kafka_config),
    "lubOilPressure": Producer(kafka_config),
    "fuelPressure": Producer(kafka_config),
    "coolantPressure": Producer(kafka_config),
    "lubOilTemp": Producer(kafka_config),
    "coolantTemp": Producer(kafka_config),
}

battery_producers = {
    "batteryVoltage": Producer(kafka_config),
    "batteryCurrent": Producer(kafka_config),
    "batteryTemperature": Producer(kafka_config),
    "batteryCapacity": Producer(kafka_config),
    "batteryResistance": Producer(kafka_config),
    "chargeCycles": Producer(kafka_config),
}

def delivery_report(err, msg):
    """Callback for message delivery reports."""
    if err is not None:
        print(f"❌ Message delivery failed: {err}")
    else:
        print(f"✅ {msg.topic()}: Data delivered successfully")

def load_battery_data():
    """Load and process battery data from Excel file"""
    try:
        # Load battery data
        battery_data = pd.read_excel(battery_excel)
        
        # Map columns to standard names
        battery_mapping = {
            'Voltage(V)': 'batteryVoltage',
            'Current(A)': 'batteryCurrent', 
            'Aux_Temperature(℃)_1': 'batteryTemperature',
            'Charge_Capacity(Ah)': 'batteryCapacity',
            'Internal Resistance(Ohm)': 'batteryResistance',
            'Cycle_Index': 'chargeCycles'
        }
        
        battery_data.rename(columns=battery_mapping, inplace=True)
        
        # Fill missing values and add some realistic variation
        battery_data = battery_data.fillna(method='ffill').fillna(method='bfill')
        
        # Add some simulated fields if missing
        if 'chargeCycles' not in battery_data.columns:
            battery_data['chargeCycles'] = np.random.randint(50, 500, len(battery_data))
        
        if 'batteryCapacity' not in battery_data.columns:
            battery_data['batteryCapacity'] = np.random.uniform(80, 95, len(battery_data))
            
        print(f"🔋 Battery dataset loaded: {len(battery_data)} samples")
        return battery_data
        
    except Exception as e:
        print(f"⚠️ Error loading battery data: {e}")
        print("🔋 Creating synthetic battery data for simulation...")
        return create_synthetic_battery_data()

def create_synthetic_battery_data(num_samples=1000):
    """Create synthetic battery data for simulation"""
    np.random.seed(42)
    
    # Simulate realistic battery parameters
    data = {
        'batteryVoltage': np.random.normal(12.6, 0.5, num_samples),  # 12V battery
        'batteryCurrent': np.random.normal(2.0, 0.8, num_samples),   # Charging current
        'batteryTemperature': np.random.normal(25, 8, num_samples),  # Temperature
        'batteryCapacity': np.random.uniform(70, 95, num_samples),   # Capacity %
        'batteryResistance': np.random.normal(0.05, 0.02, num_samples),  # Internal resistance
        'chargeCycles': np.random.randint(10, 300, num_samples)      # Charge cycles
    }
    
    return pd.DataFrame(data)

def simulate_comprehensive_vehicle_data():
    """Simulates both engine and battery sensor data streaming"""
    
    print("🚀 COMPREHENSIVE VEHICLE DATA SIMULATION STARTED")
    print("="*80)
    
    # Load engine data
    print("🚗 Loading engine sensor data...")
    engine_data = pd.read_csv(engine_csv)
    
    # Standardize engine column names
    engine_data.columns = engine_data.columns.str.strip()
    engine_column_mapping = {
        "Engine rpm": "engineRpm",
        "Lub oil pressure": "lubOilPressure", 
        "Fuel pressure": "fuelPressure",
        "Coolant pressure": "coolantPressure",
        "lub oil temp": "lubOilTemp",
        "Coolant temp": "coolantTemp",
    }
    engine_data.rename(columns=engine_column_mapping, inplace=True)
    print(f"🚗 Engine dataset loaded: {len(engine_data)} samples")
    
    # Load battery data
    battery_data = load_battery_data()
    
    print(f"\n📡 Starting real-time vehicle data streaming...")
    print(f"🚗 Engine sensors: {list(engine_producers.keys())}")
    print(f"🔋 Battery sensors: {list(battery_producers.keys())}")
    print("="*80)

    # Synchronize data streaming (cycle through both datasets)
    max_samples = min(len(engine_data), len(battery_data))
    
    for index in range(max_samples):
        timestamp = int(time.time())
        
        # Get current row data
        engine_row = engine_data.iloc[index % len(engine_data)].to_dict()
        battery_row = battery_data.iloc[index % len(battery_data)].to_dict()
        
        print(f"\n⏰ Timestamp: {timestamp} | Sample: {index + 1}/{max_samples}")
        
        # Send Engine Data
        print("🚗 Sending engine sensor data...")
        engine_missing = []
        if random.random() < 0.15:  # 15% chance of missing engine sensors
            engine_missing = random.sample(list(engine_producers.keys()), 
                                         k=random.randint(1, 2))
        
        for sensor, producer in engine_producers.items():
            if sensor in engine_missing:
                print(f"   ⏭️  Skipping {sensor}")
                continue
            
            message = {
                "sensor_id": sensor,
                "timestamp": timestamp,
                "value": float(engine_row.get(sensor, 0)),
                "vehicle_id": "VEHICLE_001",
                "sensor_type": "engine"
            }
            
            producer.produce(engine_topic, value=json.dumps(message), 
                           callback=delivery_report)
            print(f"   📤 {sensor}: {message['value']:.2f}")
        
        # Send Battery Data
        print("🔋 Sending battery sensor data...")
        battery_missing = []
        if random.random() < 0.10:  # 10% chance of missing battery sensors
            battery_missing = random.sample(list(battery_producers.keys()), 
                                          k=random.randint(1, 1))
        
        for sensor, producer in battery_producers.items():
            if sensor in battery_missing:
                print(f"   ⏭️  Skipping {sensor}")
                continue
            
            message = {
                "sensor_id": sensor,
                "timestamp": timestamp,
                "value": float(battery_row.get(sensor, 0)),
                "vehicle_id": "VEHICLE_001", 
                "sensor_type": "battery"
            }
            
            producer.produce(battery_topic, value=json.dumps(message),
                           callback=delivery_report)
            print(f"   📤 {sensor}: {message['value']:.2f}")
        
        # Flush all producers
        for producer in list(engine_producers.values()) + list(battery_producers.values()):
            producer.flush()
        
        print(f"✅ Data cycle {index + 1} completed")
        
        # Simulate real-time streaming delay
        time.sleep(3)  # 3-second intervals

    print("\n🏁 Vehicle data simulation complete!")

def simulate_engine_only():
    """Legacy function - engine data only"""
    print("🚗 Engine-only simulation mode")
    
    engine_data = pd.read_csv(engine_csv)
    engine_data.columns = engine_data.columns.str.strip()
    
    engine_column_mapping = {
        "Engine rpm": "engineRpm",
        "Lub oil pressure": "lubOilPressure",
        "Fuel pressure": "fuelPressure", 
        "Coolant pressure": "coolantPressure",
        "lub oil temp": "lubOilTemp",
        "Coolant temp": "coolantTemp",
    }
    
    engine_data.rename(columns=engine_column_mapping, inplace=True)

    for index, row in engine_data.iterrows():
        timestamp = int(time.time())
        row_dict = row.to_dict()

        # 20% chance of missing sensors
        missing_sensors = []
        if random.random() < 0.2:
            missing_sensors = random.sample(list(engine_producers.keys()), 
                                           k=random.randint(1, 2))

        for sensor, producer in engine_producers.items():
            if sensor in missing_sensors:
                continue
            
            message = {
                "sensor_id": sensor,
                "timestamp": timestamp,
                "value": row_dict.get(sensor, 0),
                "vehicle_id": "VEHICLE_001"
            }

            producer.produce('test-topic', value=json.dumps(message), 
                           callback=delivery_report)

        time.sleep(2)

if __name__ == '__main__':
    try:
        # Choose simulation mode
        print("🚀 VEHICLE DATA SIMULATION")
        print("="*50)
        print("1. Comprehensive Vehicle Data (Engine + Battery)")
        print("2. Engine Data Only (Legacy)")
        
        choice = input("\nSelect mode (1 or 2): ").strip()
        
        if choice == "1":
            simulate_comprehensive_vehicle_data()
        else:
            simulate_engine_only()
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation interrupted by user")
    finally:
        print("🔄 Flushing all producers...")
        for producer in list(engine_producers.values()) + list(battery_producers.values()):
            producer.flush()
        print("✅ All producers flushed")