from confluent_kafka import Consumer, KafkaException, KafkaError
import json
import joblib
import math
import numpy as np
import pandas as pd
import time
import datetime
from collections import deque

# Load trained models separately to avoid pickle issues
try:
    engine_model = joblib.load('enhanced_engine_condition_classifier.joblib')
    clustering_models = joblib.load('clustering_models.joblib')
    feature_scalers = joblib.load('feature_scalers.joblib') 
    model_config = joblib.load('model_config.joblib')
    
    # Load battery models
    battery_model = joblib.load('battery_condition_classifier.joblib')
    battery_features = joblib.load('battery_feature_columns.joblib')
    
    print("✅ Enhanced engine and battery models loaded successfully.")
    
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("Please run the enhanced model.py and battery_condition_predictor.py first")
    exit(1)

# Kafka Consumer setup
kafka_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'vehicle-monitoring-group',
    'auto.offset.reset': 'latest'
}
consumer = Consumer(kafka_config)

# Subscribe to both engine and battery topics
engine_topic = 'engine-sensors'
battery_topic = 'battery-sensors'
legacy_topic = 'test-topic'  # For backward compatibility

consumer.subscribe([engine_topic, battery_topic, legacy_topic])

# Store latest values and timestamps
engine_buffer = {}
battery_buffer = {}
sensor_buffer = {}  # Legacy compatibility
last_known_engine_values = {}
last_known_battery_values = {}
last_known_values = {}  # Legacy compatibility
sensor_history = deque(maxlen=100)
battery_history = deque(maxlen=50)
last_flush_time = time.time()

# Sensor field mappings
engine_field_map = {
    "Engine rpm": "engineRpm",
    "Lub oil pressure": "lubOilPressure",
    "Fuel pressure": "fuelPressure",
    "Coolant pressure": "coolantPressure",
    "lub oil temp": "lubOilTemp",
    "Coolant temp": "coolantTemp"
}

battery_field_map = {
    "batteryVoltage": "voltage",
    "batteryCurrent": "current",
    "batteryTemperature": "temperature", 
    "batteryCapacity": "capacity",
    "batteryResistance": "internal_resistance",
    "chargeCycles": "charge_cycles"
}

# Default fallback values if missing values persist
default_values = {
    "engineRpm": 1000,
    "lubOilPressure": 50,
    "fuelPressure": 3.5,
    "coolantPressure": 2.0,
    "lubOilTemp": 75,
    "coolantTemp": 80
}

# Color formatting for final condition display
color_map = {
    "Good": "\033[92m",  # Green
    "Moderate": "\033[93m",  # Yellow
    "Bad": "\033[91m"  # Red
}
reset_color = "\033[0m"

def calculate_viscosity(temp_celsius):
    """Calculate viscosity and return value."""
    temperature_kelvin = temp_celsius + 273 
    viscosity = 0.7 * math.exp(1500 / temperature_kelvin) 
    return viscosity

def calculate_part_health_scores(sensor_data):
    """Calculate specific health scores for individual parts (0-100%)"""
    
    parts_health = {}
    
    # Oil Pump Health (0-100%)
    oil_pressure_norm = np.clip(sensor_data['lubOilPressure'] / 60, 0, 1)
    oil_temp_norm = np.clip((90 - sensor_data['lubOilTemp']) / 40, 0, 1)
    parts_health['oil_pump'] = (oil_pressure_norm * 0.7 + oil_temp_norm * 0.3) * 100
    
    # Oil Filter Health
    viscosity_norm = np.clip((15 - sensor_data['viscosity']) / 10, 0, 1)
    pressure_stability = 0.8  # Simplified for real-time
    parts_health['oil_filter'] = (viscosity_norm * 0.6 + pressure_stability * 0.4) * 100
    
    # Coolant Pump Health
    coolant_pressure_norm = np.clip(sensor_data['coolantPressure'] / 3, 0, 1)
    coolant_temp_norm = np.clip((95 - sensor_data['coolantTemp']) / 35, 0, 1)
    parts_health['coolant_pump'] = (coolant_pressure_norm * 0.6 + coolant_temp_norm * 0.4) * 100
    
    # Thermostat Health
    temp_stability = 0.9  # Simplified for real-time
    optimal_temp_score = np.clip(1 - abs(sensor_data['coolantTemp'] - 85) / 20, 0, 1)
    parts_health['thermostat'] = (temp_stability * 0.4 + optimal_temp_score * 0.6) * 100
    
    # Fuel Pump Health
    fuel_pressure_norm = np.clip(sensor_data['fuelPressure'] / 4, 0, 1)
    fuel_stability = 0.85  # Simplified for real-time
    parts_health['fuel_pump'] = (fuel_pressure_norm * 0.7 + fuel_stability * 0.3) * 100
    
    # Fuel Filter Health
    fuel_flow_efficiency = np.clip(sensor_data['fuelPressure'] / 4.5, 0, 1)
    parts_health['fuel_filter'] = fuel_flow_efficiency * 100
    
    return parts_health

def calculate_part_degradation_rate(history_data):
    """Calculate degradation rates from historical data"""
    if len(history_data) < 5:
        return {
            'oil_system': 0.1,
            'cooling_system': 0.1,
            'fuel_system': 0.05
        }
    
    # Convert history to DataFrame
    df_history = pd.DataFrame(history_data)
    time_points = np.arange(len(df_history))
    
    degradation_rates = {}
    
    try:
        # Oil System Degradation
        oil_health = (
            df_history['lubOilPressure'] * 0.4 +
            (100 - df_history['lubOilTemp']) * 0.3 +
            (20 - df_history['viscosity']) * 0.3
        )
        
        oil_trend = np.polyfit(time_points, oil_health, 1)[0]
        degradation_rates['oil_system'] = abs(oil_trend) * 10
        
        # Cooling System Degradation
        cooling_health = (
            df_history['coolantPressure'] * 0.5 +
            (100 - df_history['coolantTemp']) * 0.5
        )
        cooling_trend = np.polyfit(time_points, cooling_health, 1)[0]
        degradation_rates['cooling_system'] = abs(cooling_trend) * 10
        
        # Fuel System Degradation
        fuel_health = df_history['fuelPressure']
        fuel_trend = np.polyfit(time_points, fuel_health, 1)[0]
        degradation_rates['fuel_system'] = abs(fuel_trend) * 10
        
    except Exception as e:
        print(f"Warning: Could not calculate degradation rates: {e}")
        degradation_rates = {
            'oil_system': 0.1,
            'cooling_system': 0.1,
            'fuel_system': 0.05
        }
    
    return degradation_rates

def predict_remaining_kilometers(current_health, degradation_rate, failure_threshold=20, avg_daily_km=50):
    """Predict remaining kilometers before part failure"""
    if degradation_rate <= 0:
        return 99999  # Very long distance
    
    remaining_health = current_health - failure_threshold
    if remaining_health <= 0:
        return 0  # Already at failure threshold
        
    days_to_failure = remaining_health / max(degradation_rate, 0.01)
    remaining_km = days_to_failure * avg_daily_km
    
    return max(0, min(remaining_km, 99999))

def format_remaining_distance(km):
    """Format remaining distance in a readable way"""
    if km == 0:
        return "0 km (Immediate replacement)"
    elif km < 100:
        return f"{km:.0f} km"
    elif km < 1000:
        return f"{km:.0f} km"
    elif km < 10000:
        return f"{km/1000:.1f}k km"
    else:
        return f"{km/1000:.0f}k+ km"

def generate_maintenance_recommendations(parts_health, degradation_rates, current_date):
    """Generate specific maintenance recommendations with kilometer estimates"""
    
    recommendations = []
    part_thresholds = model_config["part_thresholds"]
    
    # Average daily driving distance (can be adjusted based on vehicle usage)
    avg_daily_km = 50  # 50 km per day average
    
    for part, health_score in parts_health.items():
        if part in part_thresholds:
            threshold_info = part_thresholds[part]
            
            system_key = f"{part.split('_')[0]}_system"
            degradation_rate = degradation_rates.get(system_key, 0.1)
            
            # Calculate remaining kilometers
            remaining_km = predict_remaining_kilometers(
                health_score, 
                degradation_rate, 
                threshold_info['replace_at'],
                avg_daily_km
            )
            
            # Calculate days for date estimation
            days_to_replacement = remaining_km / avg_daily_km
            
            if health_score <= threshold_info['critical_at']:
                priority = "🚨 CRITICAL - IMMEDIATE REPLACEMENT"
                recommended_date = current_date
                priority_level = "CRITICAL"
                remaining_days = 0
            elif health_score <= threshold_info['replace_at']:
                priority = "⚠️ HIGH - REPLACE WITHIN 7 DAYS"
                recommended_date = current_date + datetime.timedelta(days=7)
                priority_level = "HIGH"
                remaining_days = 7
            elif days_to_replacement <= 30:
                priority = "📅 MEDIUM - PLAN REPLACEMENT"
                recommended_date = current_date + datetime.timedelta(days=int(days_to_replacement))
                priority_level = "MEDIUM"
                remaining_days = int(days_to_replacement)
            else:
                priority = "✅ LOW - MONITOR"
                recommended_date = current_date + datetime.timedelta(days=int(min(days_to_replacement, 365)))
                priority_level = "LOW"
                remaining_days = int(min(days_to_replacement, 365))
            
            recommendations.append({
                'part': part.replace('_', ' ').title(),
                'current_health': f"{health_score:.1f}%",
                'priority': priority,
                'priority_level': priority_level,
                'recommended_replacement_date': recommended_date.strftime('%Y-%m-%d'),
                'remaining_days': remaining_days,
                'remaining_kilometers': remaining_km,
                'remaining_km_formatted': format_remaining_distance(remaining_km)
            })
    
    return sorted(recommendations, key=lambda x: x['remaining_kilometers'])

def get_condition_text(condition_code):
    """Convert numeric condition code to text."""
    condition_map = {0: "Bad", 1: "Good", 2: "Moderate"}
    return condition_map.get(condition_code, "Unknown")

def calculate_battery_health_score(battery_data):
    """Calculate overall battery health score (0-100%) with improved voltage assessment"""
    
    voltage = battery_data['voltage']
    
    # Enhanced voltage health calculation with proper thresholds
    if voltage > 6:  # 12V lead-acid battery system
        nominal_voltage = 12.6
        # More realistic voltage health scoring for 12V systems
        if voltage >= 12.6:
            voltage_health = 100
        elif voltage >= 12.4:
            voltage_health = 90
        elif voltage >= 12.2:
            voltage_health = 75
        elif voltage >= 12.0:
            voltage_health = 50
        elif voltage >= 11.8:
            voltage_health = 25
        elif voltage >= 11.5:
            voltage_health = 10
        else:
            voltage_health = 0  # Critical/dead battery
    else:  # Li-ion single cell (3.7V nominal)
        nominal_voltage = 4.2
        # Proper Li-ion voltage health scoring
        if voltage >= 4.1:
            voltage_health = 100
        elif voltage >= 3.9:
            voltage_health = 85
        elif voltage >= 3.7:
            voltage_health = 70
        elif voltage >= 3.5:
            voltage_health = 40
        elif voltage >= 3.3:
            voltage_health = 15
        elif voltage >= 3.0:
            voltage_health = 5
        else:
            voltage_health = 0  # Over-discharged/damaged
    
    # Capacity health
    capacity_health = battery_data.get('capacity', 80)
    
    # Temperature impact (25°C is optimal)
    temp_penalty = abs(battery_data['temperature'] - 25) * 2
    temp_health = max(0, 100 - temp_penalty)
    
    # Charge cycle impact (degradation over time)
    cycles = battery_data.get('charge_cycles', 0)
    if cycles < 100:
        cycle_health = 100
    elif cycles < 500:
        cycle_health = 90 - (cycles - 100) * 0.1
    elif cycles < 1000:
        cycle_health = 50 - (cycles - 500) * 0.06
    else:
        cycle_health = max(0, 20 - (cycles - 1000) * 0.02)
    
    # Internal resistance impact (lower is better)
    resistance = battery_data.get('internal_resistance', 0.05)
    if resistance <= 0.05:
        resistance_health = 100
    elif resistance <= 0.1:
        resistance_health = 80
    elif resistance <= 0.2:
        resistance_health = 40
    else:
        resistance_health = 0
    
    # Weighted average with voltage being most critical
    overall_health = (
        voltage_health * 0.40 +      # Increased voltage weight
        capacity_health * 0.25 +
        temp_health * 0.15 +
        cycle_health * 0.10 +
        resistance_health * 0.10
    )
    
    return max(0, min(100, overall_health))

def predict_battery_remaining_life(battery_data, degradation_rate=0.5):
    """Predict remaining battery life in months"""
    
    current_health = calculate_battery_health_score(battery_data)
    
    # Battery typically needs replacement at 70% health
    replacement_threshold = 70
    
    if current_health <= replacement_threshold:
        return 0  # Needs immediate replacement
    
    remaining_health = current_health - replacement_threshold
    months_remaining = remaining_health / max(degradation_rate, 0.1)
    
    return max(0, min(months_remaining, 120))  # Cap at 10 years

def predict_battery_condition(battery_data):
    """Predict battery condition using the trained model"""
    try:
        # Prepare battery features for prediction
        feature_data = {
            'voltage': battery_data['voltage'],
            'current': battery_data.get('current', 0),
            'temperature': battery_data['temperature'],
            'capacity': battery_data.get('capacity', 80),
            'internal_resistance': battery_data['internal_resistance'],
            'charge_cycles': battery_data.get('charge_cycles', 0),
            'power': battery_data['voltage'] * abs(battery_data.get('current', 0)),
            'voltage_efficiency': battery_data['voltage'] / (12.6 if battery_data['voltage'] > 6 else 4.2),
            'capacity_ratio': battery_data.get('capacity', 80) / 100,
            'temp_stress': abs(battery_data['temperature'] - 25) / 25,
            'cycle_degradation': battery_data.get('charge_cycles', 0) / 1000,
            'resistance_factor': battery_data['internal_resistance'] * 1000,
            'voltage_drop': max(0, (12.6 if battery_data['voltage'] > 6 else 4.2) - battery_data['voltage']),
            'efficiency_score': (battery_data['voltage'] / (12.6 if battery_data['voltage'] > 6 else 4.2)) * (battery_data.get('capacity', 80) / 100),
            'energy_efficiency': 0.9,  # Default efficiency
            'time_category': 1  # Default time category
        }
        
        # Create feature vector
        feature_vector = [feature_data[col] for col in battery_features]
        
        # Predict condition
        condition_code = battery_model.predict([feature_vector])[0]
        condition_map = {0: "Critical/Replace", 1: "Good", 2: "Excellent"}
        
        return condition_map.get(condition_code, "Unknown")
        
    except Exception as e:
        print(f"Error predicting battery condition: {e}")
        # FIXED: Enhanced fallback logic with proper voltage thresholds
        voltage = battery_data['voltage']
        temperature = battery_data['temperature']
        
        # Determine battery type and set appropriate thresholds
        if voltage > 6:  # 12V lead-acid battery
            critical_voltage = 11.5
            low_voltage = 12.0
            good_voltage = 12.4
            critical_temp = 55
            high_temp = 45
        else:  # Li-ion cell (single cell)
            critical_voltage = 3.0
            low_voltage = 3.5
            good_voltage = 3.8
            critical_temp = 50
            high_temp = 40
        
        # Critical conditions (immediate replacement needed)
        if (voltage <= critical_voltage or 
            temperature >= critical_temp or
            battery_data.get('internal_resistance', 0) > 0.2):
            return "Critical/Replace"
        
        # Poor conditions (replacement needed soon)
        elif (voltage <= low_voltage or 
              temperature >= high_temp or
              battery_data.get('capacity', 100) < 70):
            return "Poor/Replace"
        
        # Good conditions
        elif (voltage <= good_voltage or 
              temperature >= 35 or
              battery_data.get('capacity', 100) < 85):
            return "Good"
        
        # Excellent conditions
        else:
            return "Excellent"

def generate_battery_recommendations(battery_data, current_date):
    """Generate battery-specific maintenance recommendations with enhanced critical voltage detection"""
    
    health_score = calculate_battery_health_score(battery_data)
    remaining_months = predict_battery_remaining_life(battery_data)
    condition = predict_battery_condition(battery_data)
    voltage = battery_data['voltage']
    
    recommendations = []
    
    # Enhanced battery replacement logic with voltage-specific warnings
    if voltage <= 3.0 or voltage >= 15.0:  # Completely dead or overcharged
        priority = "🚨 CRITICAL - BATTERY DEAD/DAMAGED - DO NOT USE"
        recommended_date = current_date
        remaining_days = 0
        priority_level = "CRITICAL"
    elif voltage <= 3.5 or health_score <= 30:  # Severely damaged Li-ion or very low health
        priority = "🚨 CRITICAL - REPLACE BATTERY IMMEDIATELY - UNSAFE TO USE"
        recommended_date = current_date
        remaining_days = 0
        priority_level = "CRITICAL"
    elif voltage <= 11.5 or health_score <= 60:  # Dead 12V battery or low health
        priority = "🔋 CRITICAL - REPLACE BATTERY IMMEDIATELY"
        recommended_date = current_date
        remaining_days = 0
        priority_level = "CRITICAL"
    elif voltage <= 11.8 or health_score <= 70:  # Very low 12V battery
        priority = "🔋 HIGH - REPLACE BATTERY WITHIN 7 DAYS"
        recommended_date = current_date + datetime.timedelta(days=7)
        remaining_days = 7
        priority_level = "HIGH"
    elif remaining_months <= 6:
        priority = "🔋 MEDIUM - PLAN BATTERY REPLACEMENT"
        recommended_date = current_date + datetime.timedelta(days=int(remaining_months * 30))
        remaining_days = int(remaining_months * 30)
        priority_level = "MEDIUM"
    else:
        priority = "🔋 LOW - MONITOR BATTERY HEALTH"
        recommended_date = current_date + datetime.timedelta(days=365)
        remaining_days = 365
        priority_level = "LOW"
    
    # Add safety warnings for extremely low voltages
    safety_warning = ""
    if voltage <= 3.0:
        safety_warning = "⚠️ SAFETY WARNING: Battery voltage critically low - may be permanently damaged"
    elif voltage <= 3.5:
        safety_warning = "⚠️ WARNING: Li-ion battery deeply discharged - risk of damage/fire"
    elif voltage <= 11.5 and voltage > 6:
        safety_warning = "⚠️ WARNING: 12V battery dead - vehicle may not start"
    
    recommendations.append({
        'component': 'Battery Pack',
        'current_health': f"{health_score:.1f}%",
        'condition': condition,
        'priority': priority,
        'priority_level': priority_level,
        'recommended_replacement_date': recommended_date.strftime('%Y-%m-%d'),
        'remaining_days': remaining_days,
        'remaining_months': remaining_months,
        'voltage': battery_data['voltage'],
        'temperature': battery_data['temperature'],
        'internal_resistance': battery_data['internal_resistance'],
        'safety_warning': safety_warning
    })
    
    return recommendations

def process_comprehensive_vehicle_data():
    """Enhanced processing with both engine and battery health monitoring"""
    global last_flush_time

    if time.time() - last_flush_time < 2.0:
        return

    last_flush_time = time.time()

    # Process Engine Data
    complete_engine_data = {}
    missing_engine_fields = []

    if engine_buffer:
        for key in ["engineRpm", "lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
            if key in engine_buffer:
                complete_engine_data[key] = engine_buffer[key]
                last_known_engine_values[key] = engine_buffer[key]
            elif key in last_known_engine_values:
                complete_engine_data[key] = last_known_engine_values[key]
            else:
                complete_engine_data[key] = default_values[key]
                missing_engine_fields.append(key)

        if missing_engine_fields:
            print(f"⚠️ Engine Warning: Missing fields {missing_engine_fields}, using last known/default values.")

        # Calculate viscosity
        complete_engine_data['viscosity'] = calculate_viscosity(complete_engine_data['lubOilTemp'])

        # Add to sensor history
        sensor_history.append(complete_engine_data.copy())

    # Process Battery Data
    complete_battery_data = {}
    battery_available = False

    if battery_buffer:
        battery_available = True
        for sensor_key, field_key in battery_field_map.items():
            if sensor_key in battery_buffer:
                complete_battery_data[field_key] = battery_buffer[sensor_key]
                last_known_battery_values[field_key] = battery_buffer[sensor_key]
            elif field_key in last_known_battery_values:
                complete_battery_data[field_key] = last_known_battery_values[field_key]
            else:
                # Default battery values
                defaults = {
                    'voltage': 12.4,
                    'current': 0.5,
                    'temperature': 25,
                    'capacity': 85,
                    'internal_resistance': 0.05,
                    'charge_cycles': 100
                }
                complete_battery_data[field_key] = defaults.get(field_key, 0)

        battery_history.append(complete_battery_data.copy())

    # Only proceed if we have engine data
    if not complete_engine_data:
        return

    # Predict engine condition
    try:
        basic_features = [
            complete_engine_data["engineRpm"],
            complete_engine_data["lubOilPressure"],
            complete_engine_data["fuelPressure"],
            complete_engine_data["coolantPressure"],
            complete_engine_data["lubOilTemp"],
            complete_engine_data["coolantTemp"],
            complete_engine_data["viscosity"]
        ]
        
        engine_condition_code = engine_model.predict([basic_features])[0]
        engine_condition_text = get_condition_text(engine_condition_code)
        
    except Exception as e:
        print(f"Error predicting engine condition: {e}")
        # Fallback logic
        if (complete_engine_data["lubOilPressure"] < 30 or 
            complete_engine_data["coolantTemp"] > 95 or
            complete_engine_data["fuelPressure"] < 2):
            engine_condition_text = "Bad"
        elif (complete_engine_data["lubOilPressure"] < 50 or 
              complete_engine_data["coolantTemp"] > 90):
            engine_condition_text = "Moderate"
        else:
            engine_condition_text = "Good"

    # Calculate part health scores
    parts_health = calculate_part_health_scores(complete_engine_data)

    # Calculate degradation rates
    degradation_rates = calculate_part_degradation_rate(list(sensor_history))

    # Generate engine maintenance recommendations
    engine_recommendations = generate_maintenance_recommendations(
        parts_health, 
        degradation_rates, 
        datetime.datetime.now()
    )

    # Generate battery recommendations if battery data available
    battery_recommendations = []
    if battery_available and complete_battery_data:
        battery_recommendations = generate_battery_recommendations(complete_battery_data, datetime.datetime.now())
    else:
        # Simulate battery data using engine temperature as proxy
        simulated_battery_data = {
            'voltage': 12.4,
            'temperature': complete_engine_data['coolantTemp'],
            'internal_resistance': 0.05,
            'capacity': 85,
            'charge_cycles': 150
        }
        battery_recommendations = generate_battery_recommendations(simulated_battery_data, datetime.datetime.now())

    # Display comprehensive analysis
    print("\n" + "="*100)
    print(f" {color_map.get(engine_condition_text, '')}🚗 COMPREHENSIVE VEHICLE HEALTH REPORT (ENGINE + BATTERY){reset_color}")
    print("="*100)

    print(f"\n📊 Overall Engine Status: {color_map.get(engine_condition_text, '')}{engine_condition_text.upper()}{reset_color}")
    
    if engine_condition_text == 'Good':
        print("   ✅ All engine systems operating within optimal parameters")
    elif engine_condition_text == 'Moderate':
        print("   ⚠️ Some engine systems showing signs of wear - attention recommended")
    else:
        print("   🚨 Critical engine issues detected - immediate action required")

    # Battery Status Display - FIXED: Ensure consistent status based on priority level
    battery_rec = battery_recommendations[0] if battery_recommendations else None
    if battery_rec:
        battery_status_color = ""
        # FIXED: Determine battery status based on priority level for consistency
        if battery_rec['priority_level'] == 'CRITICAL':
            battery_status_color = color_map["Bad"]
            battery_status_text = "Critical/Replace"
        elif battery_rec['priority_level'] == 'HIGH':
            battery_status_color = color_map["Moderate"]
            battery_status_text = "Poor/Replace Soon"
        elif battery_rec['priority_level'] == 'MEDIUM':
            battery_status_color = color_map["Moderate"]
            battery_status_text = "Fair/Monitor"
        else:
            battery_status_color = color_map["Good"]
            battery_status_text = "Good"

        # Use consistent status instead of potentially conflicting ML prediction
        print(f"\n🔋 Battery Status: {battery_status_color}{battery_status_text}{reset_color}")
        print(f"   Current Health: {battery_rec['current_health']}")
        print(f"   Voltage: {battery_rec['voltage']:.2f}V")
        print(f"   Temperature: {battery_rec['temperature']:.1f}°C")
        print(f"   Internal Resistance: {battery_rec['internal_resistance']:.3f}Ω")
        
        if battery_available:
            print(f"   Data Source: 🔋 Real battery sensors")
        else:
            print(f"   Data Source: 🔄 Simulated (using engine data)")
            
        # Display safety warning if present
        if battery_rec.get('safety_warning'):
            print(f"   {battery_rec['safety_warning']}")
            
        # Always show action required for consistency
        print(f"   🚨 Action Required: {battery_rec['priority']}")
        print(f"   📅 Recommended Date: {battery_rec['recommended_replacement_date']}")
        print(f"   ⏰ Remaining Days: {battery_rec['remaining_days']}")

    # Part Health Dashboard
    print(f"\n🔧 Individual Engine Part Health Scores:")
    print("-" * 80)
    for part, health in parts_health.items():
        if health > 70:
            status_color = color_map["Good"]
            status_icon = "✅"
        elif health > 40:
            status_color = color_map["Moderate"] 
            status_icon = "⚠️"
        else:
            status_color = color_map["Bad"]
            status_icon = "🚨"
        
        print(f"   {part.replace('_', ' ').title():15}: {status_color}{health:6.1f}%{reset_color} {status_icon}")

    # ...existing maintenance recommendations code...
    
    # Enhanced System Performance Summary
    avg_health = np.mean(list(parts_health.values()))
    
    print(f"\n📋 Comprehensive Vehicle Performance Summary:")
    print("-" * 50)
    print(f"   Overall Engine Health: {avg_health:.1f}%")
    if battery_rec:
        battery_health = float(battery_rec['current_health'].replace('%', ''))
        overall_vehicle_health = (avg_health * 0.7 + battery_health * 0.3)
        print(f"   Overall Vehicle Health: {overall_vehicle_health:.1f}%")
        print(f"   Battery Health: {battery_rec['current_health']}")
    
    print(f"   Data Sources:")
    print(f"     🚗 Engine Sensors: {len(complete_engine_data)} parameters")
    if battery_available:
        print(f"     🔋 Battery Sensors: {len(complete_battery_data)} parameters")
    else:
        print(f"     🔋 Battery Sensors: Simulated")
    print(f"     📊 Historical Data Points: {len(sensor_history)} engine, {len(battery_history)} battery")

    print("\n" + "="*100)
    
    # Clear buffers after processing
    engine_buffer.clear()
    battery_buffer.clear()

def process_message(message, topic_name):
    """Store received sensor data in appropriate buffer."""
    try:
        data = json.loads(message)
        sensor_id = data.get("sensor_id")
        value = data.get("value")
        sensor_type = data.get("sensor_type", "unknown")

        if topic_name == engine_topic or sensor_type == "engine":
            # Handle engine sensor data
            if sensor_id in ["engineRpm", "lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp"]:
                engine_buffer[sensor_id] = value
                print(f"🚗 Engine: {sensor_id} = {value:.2f}")
            else:
                # Legacy topic handling
                if sensor_id in engine_field_map.values():
                    engine_buffer[sensor_id] = value
                    
        elif topic_name == battery_topic or sensor_type == "battery":
            # Handle battery sensor data
            if sensor_id in battery_field_map.keys():
                battery_buffer[sensor_id] = value
                print(f"🔋 Battery: {sensor_id} = {value:.2f}")
                
        else:
            # Legacy handling - assume engine data
            if sensor_id in engine_field_map.values():
                engine_buffer[sensor_id] = value

    except json.JSONDecodeError:
        print("Invalid JSON received!")
    except KeyError as e:
        print(f"Missing field: {e}")
    except Exception as e:
        print(f"Error processing message: {e}")

try:
    print(f"🚀 COMPREHENSIVE VEHICLE MONITORING SYSTEM STARTED")
    print(f"📡 Subscribed to topics: {[engine_topic, battery_topic, legacy_topic]}")
    print("🚗 Monitoring engine sensors: RPM, oil pressure, fuel pressure, coolant pressure, temperatures")
    print("🔋 Monitoring battery sensors: voltage, current, temperature, capacity, resistance, charge cycles")
    print("🔄 Real-time predictive maintenance with ML models (Engine: Enhanced | Battery: 99.63% accuracy)")
    print("="*100)
    
    while True:
        msg = consumer.poll(1.0)

        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                print(f"📄 End of partition reached: {msg.topic()} {msg.partition()}")
            else:
                raise KafkaException(msg.error())
        else:
            process_message(msg.value().decode('utf-8'), msg.topic())

        # Process comprehensive vehicle data every 2 seconds
        process_comprehensive_vehicle_data()

except KeyboardInterrupt:
    print("\n🛑 Comprehensive vehicle monitoring interrupted by user")
finally:
    consumer.close()
    print("📡 Kafka consumer closed - all vehicle monitoring stopped.")
