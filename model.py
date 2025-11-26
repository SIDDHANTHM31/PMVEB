import pandas as pd
import joblib
import math
import numpy as np
import datetime
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data_path = "engines_dataset-train_multi_class.csv" 
df = pd.read_csv(data_path)

# Define expected field names and map them to correct ones
field_map = {
    "Engine rpm": "engineRpm",
    "Lub oil pressure": "lubOilPressure",
    "Fuel pressure": "fuelPressure",
    "Coolant pressure": "coolantPressure",
    "lub oil temp": "lubOilTemp",
    "Coolant temp": "coolantTemp",
    "Engine Condition": "engineCondition" 
}

df.rename(columns=field_map, inplace=True)

# Add timestamps if not present (for demonstration)
if 'timestamp' not in df.columns:
    df['timestamp'] = pd.date_range(start='2024-01-01', periods=len(df), freq='H')

# ===== ENHANCED FEATURE ENGINEERING FUNCTIONS =====

def calculate_viscosity(temp_celsius):
    """Calculate viscosity from temperature using empirical formula"""
    temperature_kelvin = temp_celsius + 273  # Convert °C to K
    return 0.7 * math.exp(1500 / temperature_kelvin)  # Empirical formula

def analyze_pressure_patterns(pressure_data, param_name):
    """Advanced analysis for pressure parameters"""
    pressure_data = np.array(pressure_data)
    
    # Rate of change analysis
    pressure_gradient = np.gradient(pressure_data)
    
    # Pressure stability (rolling standard deviation)
    window_size = min(10, len(pressure_data) // 4)
    if window_size > 1:
        pressure_stability = pd.Series(pressure_data).rolling(window=window_size).std().fillna(0).values
    else:
        pressure_stability = np.zeros_like(pressure_data)
    
    # Pressure efficiency ratio (current / max optimal for parameter)
    optimal_ranges = {
        "lubOilPressure": 60,    # Optimal lub oil pressure
        "fuelPressure": 4,       # Optimal fuel pressure
        "coolantPressure": 3     # Optimal coolant pressure
    }
    pressure_efficiency = pressure_data / optimal_ranges.get(param_name, np.max(pressure_data))
    
    return np.column_stack([
        pressure_data,
        pressure_gradient,
        pressure_stability,
        pressure_efficiency
    ])

def analyze_temperature_patterns(temp_data, param_name):
    """Advanced analysis for temperature parameters"""
    temp_data = np.array(temp_data)
    
    # Temperature rate of change
    temp_gradient = np.gradient(temp_data)
    
    # Temperature stability analysis
    window_size = min(10, len(temp_data) // 4)
    if window_size > 1:
        temp_stability = pd.Series(temp_data).rolling(window=window_size).std().fillna(0).values
    else:
        temp_stability = np.zeros_like(temp_data)
    
    # Operating efficiency zones based on optimal ranges
    optimal_ranges = {
        "lubOilTemp": 75,    # Optimal oil temperature
        "coolantTemp": 85    # Optimal coolant temperature
    }
    
    optimal_temp = optimal_ranges.get(param_name, np.mean(temp_data))
    temp_deviation = np.abs(temp_data - optimal_temp)
    
    # For oil temperature, also calculate thermal efficiency
    if param_name == "lubOilTemp":
        thermal_efficiency = np.exp(-temp_deviation / optimal_temp)  # Exponential decay from optimal
    else:
        thermal_efficiency = 1 / (1 + temp_deviation / optimal_temp)  # Inverse relationship
    
    return np.column_stack([
        temp_data,
        temp_gradient,
        temp_stability,
        temp_deviation,
        thermal_efficiency
    ])

def analyze_rpm_patterns(rpm_data, param_name=None):
    """Advanced RPM analysis with load zones"""
    rpm_data = np.array(rpm_data)
    
    # RPM normalization
    rpm_max = np.max(rpm_data)
    rpm_normalized = rpm_data / rpm_max
    
    # RPM variability analysis
    window_size = min(10, len(rpm_data) // 4)
    if window_size > 1:
        rpm_variability = pd.Series(rpm_data).rolling(window=window_size).std().fillna(0).values
    else:
        rpm_variability = np.zeros_like(rpm_data)
    
    # Load zone classification
    rpm_load_zones = np.zeros_like(rpm_data)
    rpm_load_zones[rpm_data < 1500] = 0  # Idle
    rpm_load_zones[(rpm_data >= 1500) & (rpm_data < 2500)] = 1  # Normal
    rpm_load_zones[(rpm_data >= 2500) & (rpm_data < 3500)] = 2  # High Load
    rpm_load_zones[rpm_data >= 3500] = 3  # Critical Load
    
    # RPM efficiency (optimal around 2000-2500 RPM)
    rpm_efficiency = np.exp(-np.abs(rpm_data - 2250) / 2250)
    
    return np.column_stack([
        rpm_data,
        rpm_normalized,
        rpm_variability,
        rpm_load_zones,
        rpm_efficiency
    ])

def calculate_part_health_scores(sensor_data):
    """Calculate specific health scores for individual parts (0-100%)"""
    
    parts_health = {}
    
    # Oil Pump Health (0-100%)
    oil_pressure_norm = np.clip(sensor_data['lubOilPressure'] / 60, 0, 1)  # 60 is optimal
    oil_temp_norm = np.clip((90 - sensor_data['lubOilTemp']) / 40, 0, 1)   # 50-90 is good range
    parts_health['oil_pump'] = (oil_pressure_norm * 0.7 + oil_temp_norm * 0.3) * 100
    
    # Oil Filter Health (based on viscosity and pressure drop)
    viscosity_norm = np.clip((15 - sensor_data['viscosity']) / 10, 0, 1)   # Lower viscosity is better
    pressure_stability = 1 - min(np.std(sensor_data['lubOilPressure']) / max(np.mean(sensor_data['lubOilPressure']), 1), 1)
    parts_health['oil_filter'] = (viscosity_norm * 0.6 + pressure_stability * 0.4) * 100
    
    # Coolant Pump Health
    coolant_pressure_norm = np.clip(sensor_data['coolantPressure'] / 3, 0, 1)
    coolant_temp_norm = np.clip((95 - sensor_data['coolantTemp']) / 35, 0, 1)
    parts_health['coolant_pump'] = (coolant_pressure_norm * 0.6 + coolant_temp_norm * 0.4) * 100
    
    # Thermostat Health (based on temperature stability)
    temp_stability = 1 - min(np.std(sensor_data['coolantTemp']) / max(np.mean(sensor_data['coolantTemp']), 1), 1)
    optimal_temp_score = np.clip(1 - abs(sensor_data['coolantTemp'] - 85) / 20, 0, 1)
    parts_health['thermostat'] = (temp_stability * 0.4 + optimal_temp_score * 0.6) * 100
    
    # Fuel Pump Health
    fuel_pressure_norm = np.clip(sensor_data['fuelPressure'] / 4, 0, 1)
    fuel_stability = 1 - min(np.std(sensor_data['fuelPressure']) / max(np.mean(sensor_data['fuelPressure']), 1), 1)
    parts_health['fuel_pump'] = (fuel_pressure_norm * 0.7 + fuel_stability * 0.3) * 100
    
    # Fuel Filter Health (based on pressure consistency)
    fuel_flow_efficiency = np.clip(sensor_data['fuelPressure'] / 4.5, 0, 1)  # 4.5 is optimal
    parts_health['fuel_filter'] = fuel_flow_efficiency * 100
    
    return parts_health

# ===== COMPUTE ADVANCED FEATURES =====

print("Computing advanced features for all parameters...")

# Compute viscosity (existing feature)
df["viscosity"] = df["lubOilTemp"].apply(calculate_viscosity)

# Calculate part health scores for the dataset
print("Calculating part health scores for entire dataset...")
parts_health_dataset = {}

for idx, row in df.iterrows():
    sensor_data = {
        'lubOilPressure': row['lubOilPressure'],
        'lubOilTemp': row['lubOilTemp'],
        'coolantPressure': row['coolantPressure'], 
        'coolantTemp': row['coolantTemp'],
        'fuelPressure': row['fuelPressure'],
        'viscosity': row['viscosity']
    }
    
    parts_health = calculate_part_health_scores(sensor_data)
    
    for part, health in parts_health.items():
        if part not in parts_health_dataset:
            parts_health_dataset[part] = []
        parts_health_dataset[part].append(health)

# Add part health scores to dataframe
for part, health_scores in parts_health_dataset.items():
    df[f"{part}_health"] = health_scores

print(f"Added health scores for {len(parts_health_dataset)} parts to the dataset.")

# ===== MULTI-PARAMETER CLUSTERING ANALYSIS =====

parameters_for_clustering = {
    "lubOilPressure": {
        "clusters": 3, 
        "analysis_function": analyze_pressure_patterns,
        "feature_names": ["pressure", "gradient", "stability", "efficiency"]
    },
    "fuelPressure": {
        "clusters": 3, 
        "analysis_function": analyze_pressure_patterns,
        "feature_names": ["pressure", "gradient", "stability", "efficiency"]
    },
    "coolantPressure": {
        "clusters": 3, 
        "analysis_function": analyze_pressure_patterns,
        "feature_names": ["pressure", "gradient", "stability", "efficiency"]
    },
    "lubOilTemp": {
        "clusters": 3, 
        "analysis_function": analyze_temperature_patterns,
        "feature_names": ["temperature", "gradient", "stability", "deviation", "efficiency"]
    },
    "coolantTemp": {
        "clusters": 3, 
        "analysis_function": analyze_temperature_patterns,
        "feature_names": ["temperature", "gradient", "stability", "deviation", "efficiency"]
    },
    "engineRpm": {
        "clusters": 4, 
        "analysis_function": analyze_rpm_patterns,
        "feature_names": ["rpm", "normalized", "variability", "load_zones", "efficiency"]
    },
    "viscosity": {
        "clusters": 3, 
        "analysis_function": None,
        "feature_names": ["viscosity"]
    }
}

# Store all clustering models and scalers
clustering_models = {}
feature_scalers = {}

print("\n" + "="*80)
print("MULTI-PARAMETER CLUSTERING ANALYSIS")
print("="*80)

for param, config in parameters_for_clustering.items():
    print(f"\nAnalyzing parameter: {param}")
    
    if config["analysis_function"] is not None:
        if param == "viscosity":
            features = df[param].values.reshape(-1, 1)
        else:
            features = config["analysis_function"](df[param], param)
    else:
        features = df[param].values.reshape(-1, 1)
    
    # Apply feature scaling for better clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Determine optimal number of clusters using silhouette analysis
    silhouette_scores = []
    K_range = range(2, min(6, len(df) // 10))
    
    if len(K_range) > 0:
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans_temp.fit_predict(features_scaled)
            silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        best_k = K_range[np.argmax(silhouette_scores)]
        optimal_clusters = config["clusters"] if config["clusters"] in K_range else best_k
        print(f"  Optimal clusters: {optimal_clusters}")
        print(f"  Best silhouette score: {max(silhouette_scores):.3f}")
    else:
        optimal_clusters = config["clusters"]
        print(f"  Using configured clusters: {optimal_clusters}")
    
    # Train final K-Means clustering model
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    kmeans.fit(features_scaled)
    
    # Get cluster assignments
    cluster_assignments = kmeans.predict(features_scaled)
    
    # Store models (simplified structure)
    clustering_models[param] = kmeans
    feature_scalers[param] = {
        "scaler": scaler,
        "feature_names": config["feature_names"]
    }
    
    # Add cluster assignments to dataframe
    df[f"{param}_cluster"] = cluster_assignments
    
    # Print cluster statistics
    unique_clusters = np.unique(cluster_assignments)
    print(f"  Cluster distribution:")
    for cluster in unique_clusters:
        cluster_size = np.sum(cluster_assignments == cluster)
        cluster_percentage = (cluster_size / len(cluster_assignments)) * 100
        print(f"    Cluster {cluster}: {cluster_size} samples ({cluster_percentage:.1f}%)")

# ===== TRAIN ENHANCED ENGINE CONDITION CLASSIFIER =====

print("\n" + "="*80)
print("TRAINING ENHANCED ENGINE CONDITION CLASSIFIER")
print("="*80)

# Enhanced feature set including cluster information and part health scores
feature_columns = ["engineRpm", "lubOilPressure", "fuelPressure", "coolantPressure", "lubOilTemp", "coolantTemp", "viscosity"]

# Add cluster features
cluster_columns = [col for col in df.columns if col.endswith('_cluster')]
feature_columns.extend(cluster_columns)

# Add part health features
health_columns = [col for col in df.columns if col.endswith('_health')]
feature_columns.extend(health_columns)

X = df[feature_columns]
y = df["engineCondition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train enhanced RandomForest for engine condition
model = RandomForestClassifier(
    n_estimators=300, 
    random_state=42, 
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Enhanced Engine Condition Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

# ===== SAVE MODELS SEPARATELY (FIX FOR PICKLE ISSUE) =====
print(f"\nSaving models...")

# Save main models separately to avoid pickle issues
joblib.dump(model, "enhanced_engine_condition_classifier.joblib")
joblib.dump(clustering_models, "clustering_models.joblib")
joblib.dump(feature_scalers, "feature_scalers.joblib")

# Save a simple configuration dictionary instead of functions
model_config = {
    "feature_columns": feature_columns,
    "part_thresholds": {
        'oil_filter': {'replace_at': 30, 'critical_at': 15},
        'fuel_filter': {'replace_at': 25, 'critical_at': 10},
        'oil_pump': {'replace_at': 40, 'critical_at': 20},
        'coolant_pump': {'replace_at': 35, 'critical_at': 15},
        'thermostat': {'replace_at': 45, 'critical_at': 25},
        'fuel_pump': {'replace_at': 40, 'critical_at': 20}
    },
    "base_costs": {
        'oil_filter': {'replacement': 25, 'failure': 500},
        'fuel_filter': {'replacement': 35, 'failure': 800},
        'oil_pump': {'replacement': 300, 'failure': 2500},
        'coolant_pump': {'replacement': 250, 'failure': 3000},
        'thermostat': {'replacement': 50, 'failure': 1200},
        'fuel_pump': {'replacement': 200, 'failure': 1800}
    }
}

joblib.dump(model_config, "model_config.joblib")

print("Models saved successfully!")
print("Generated files:")
print("- enhanced_engine_condition_classifier.joblib")
print("- clustering_models.joblib") 
print("- feature_scalers.joblib")
print("- model_config.joblib")

print("\n" + "="*80)
print("ENHANCED MODEL TRAINING COMPLETE!")
print("="*80)




# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import warnings
# warnings.filterwarnings("ignore")

# print("="*100)
# print("PREDICTIVE ENGINE MAINTENANCE: END-TO-END TRAINING (Preprocessing + SMOTE + XGBoost)")
# print("="*100)

# # ============================
# # Load Dataset
# # ============================
# df = pd.read_excel("Engine_data.xlsx")   # make sure file is in same folder
# print("Columns in dataset:", list(df.columns))
# print("Dataset loaded with shape:", df.shape)

# # ============================
# # Define Features and Target
# # ============================
# sensors = [
#     "Engine rpm",
#     "Lub oil pressure",
#     "Fuel pressure",
#     "Coolant pressure",
#     "lubricant oil temperature",
#     "Coolant temperature"
# ]
# target = "Engine Condition"


# # ============================
# # Preprocessing
# # ============================
# X = df[sensors]
# y = df[target]

# # Normalize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Handle imbalance with SMOTE
# smote = SMOTE(random_state=42)
# X_res, y_res = smote.fit_resample(X_scaled, y)

# print("After SMOTE, dataset shape:", X_res.shape, y_res.shape)

# # ============================
# # Train/Test Split
# # ============================
# X_train, X_test, y_train, y_test = train_test_split(
#     X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
# )

# # ============================
# # XGBoost Classifier
# # ============================
# model = XGBClassifier(
#     n_estimators=200,
#     learning_rate=0.1,
#     max_depth=5,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42,
#     eval_metric="mlogloss"
# )

# # Cross-validation
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
# print("Cross-validation accuracy scores:", cv_scores)
# print("Mean CV accuracy:", np.mean(cv_scores))

# # Train model
# model.fit(X_train, y_train)

# # ============================
# # Evaluation
# # ============================
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("\nTest Accuracy:", acc)
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

# # Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=np.unique(y), yticklabels=np.unique(y))
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# # plt.show()

# # ============================
# # Feature Importance
# # ============================
# importance = model.feature_importances_
# # plt.figure(figsize=(8, 5))
# # sns.barplot(x=importance, y=sensors)
# # plt.title("Feature Importance (XGBoost)")
# # plt.show()
