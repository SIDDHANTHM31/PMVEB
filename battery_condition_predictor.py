import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class BatteryConditionPredictor:
    def __init__(self, data_path="charging_experiment_7_Channel_7.1 (1).xlsx", random_state=42):
        self.data_path = data_path
        self.random_state = random_state
        self.model = None
        self.scaler = None
        
    def load_and_analyze_battery_data(self):
        """Load and analyze battery charging experiment data"""
        print("="*80)
        print("BATTERY CONDITION PREDICTOR - DATA ANALYSIS")
        print("="*80)
        
        try:
            # Try to read the Excel file
            df = pd.read_excel(self.data_path)
            print(f"Battery dataset loaded successfully!")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            print("Creating synthetic battery data for demonstration...")
            df = self.create_synthetic_battery_data()
        
        return df
    
    def create_synthetic_battery_data(self):
        """Create synthetic battery data based on typical battery parameters"""
        print("Creating synthetic battery charging experiment data...")
        
        np.random.seed(self.random_state)
        n_samples = 5000
        
        # Battery parameters
        voltage = np.random.normal(12.6, 0.8, n_samples)  # 12V battery voltage
        current = np.random.normal(5.0, 1.5, n_samples)   # Charging current (A)
        temperature = np.random.normal(25, 10, n_samples)  # Temperature (°C)
        capacity = np.random.normal(75, 15, n_samples)     # State of Charge (%)
        internal_resistance = np.random.normal(0.02, 0.01, n_samples)  # Internal resistance (Ohms)
        charge_cycles = np.random.randint(0, 2000, n_samples)  # Number of charge cycles
        
        # Time-based features
        time_hours = np.random.uniform(0, 24, n_samples)
        
        # Create battery condition based on multiple factors
        battery_health = 100 - (charge_cycles / 20) - np.abs(temperature - 25) * 0.5
        battery_health += np.random.normal(0, 5, n_samples)  # Add noise
        battery_health = np.clip(battery_health, 0, 100)
        
        # Define condition categories
        conditions = []
        for health in battery_health:
            if health >= 80:
                conditions.append(2)  # Excellent
            elif health >= 60:
                conditions.append(1)  # Good
            else:
                conditions.append(0)  # Poor/Replace
        
        df = pd.DataFrame({
            'voltage': voltage,
            'current': current,
            'temperature': temperature,
            'capacity': capacity,
            'internal_resistance': internal_resistance,
            'charge_cycles': charge_cycles,
            'time_hours': time_hours,
            'battery_health': battery_health,
            'battery_condition': conditions
        })
        
        print(f"Synthetic battery dataset created with {len(df)} samples")
        
        return df
    
    def engineer_battery_features(self, df):
        """Create advanced battery features"""
        print("\nEngineering battery features...")
        
        # Map actual column names to standardized names
        column_mapping = {
            'Voltage(V)': 'voltage',
            'Current(A)': 'current', 
            'Aux_Temperature(℃)_1': 'temperature',
            'Charge_Capacity(Ah)': 'capacity',
            'Internal Resistance(Ohm)': 'internal_resistance',
            'Cycle_Index': 'charge_cycles',
            'Test_Time(s)': 'test_time',
            'ACR(Ohm)': 'acr',
            'Charge_Energy(Wh)': 'charge_energy',
            'Discharge_Energy(Wh)': 'discharge_energy'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Fill missing values
        df = df.ffill().bfill()
        
        # Power calculation
        df['power'] = df['voltage'] * abs(df['current'])
        
        # Energy efficiency metrics
        df['voltage_efficiency'] = df['voltage'] / 4.2  # Normalized to nominal 4.2V (Li-ion)
        df['capacity_ratio'] = df['capacity'] / df['capacity'].max()  # Normalized capacity
        
        # Temperature impact (25°C is optimal)
        df['temp_stress'] = np.abs(df['temperature'] - 25) / 25
        
        # Aging factors
        df['cycle_degradation'] = df['charge_cycles'] / df['charge_cycles'].max()  # Normalized cycles
        df['resistance_factor'] = df['internal_resistance'] * 1000  # Convert to mΩ
        
        # Battery health indicators
        df['voltage_drop'] = np.maximum(0, 4.2 - df['voltage'])  # 4.2V is full charge for Li-ion
        df['efficiency_score'] = df['voltage_efficiency'] * df['capacity_ratio'] * (1 - df['temp_stress'])
        
        # Energy efficiency
        df['energy_efficiency'] = df['charge_energy'] / (df['discharge_energy'] + 0.001)
        
        # Time-based features
        df['time_hours'] = df['test_time'] / 3600  # Convert seconds to hours
        df['time_category'] = pd.cut(df['time_hours'] % 24, 
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=[0, 1, 2, 3]).astype(int)
        
        # Create battery condition based on multiple factors
        # Calculate battery health percentage
        voltage_factor = (df['voltage'] / 4.2) * 100
        capacity_factor = (df['capacity'] / df['capacity'].max()) * 100
        resistance_factor = np.clip(100 - (df['internal_resistance'] * 10000), 0, 100)
        temp_factor = np.clip(100 - np.abs(df['temperature'] - 25) * 2, 0, 100)
        cycle_factor = np.clip(100 - (df['charge_cycles'] / 50), 0, 100)
        
        df['battery_health'] = (
            voltage_factor * 0.25 +
            capacity_factor * 0.30 +
            resistance_factor * 0.20 +
            temp_factor * 0.15 +
            cycle_factor * 0.10
        )
        
        # Define condition categories based on battery health
        df['battery_condition'] = pd.cut(df['battery_health'], 
                                       bins=[0, 60, 80, 100], 
                                       labels=[0, 1, 2]).astype(int)  # 0=Poor, 1=Good, 2=Excellent
        
        print(f"Feature engineering complete. Total features: {len(df.columns)}")
        
        return df
    
    def train_battery_model(self, df):
        """Train battery condition prediction model"""
        print("\nTraining battery condition model...")
        
        # Prepare features
        feature_columns = [
            'voltage', 'current', 'temperature', 'capacity', 'internal_resistance',
            'charge_cycles', 'power', 'voltage_efficiency', 'capacity_ratio',
            'temp_stress', 'cycle_degradation', 'resistance_factor',
            'voltage_drop', 'efficiency_score', 'energy_efficiency', 'time_category'
        ]
        
        X = df[feature_columns]
        y = df['battery_condition']
        
        print(f"Features used: {feature_columns}")
        print(f"Class distribution:")
        class_dist = y.value_counts(normalize=True) * 100
        for idx, val in class_dist.items():
            print(f"  Class {idx}: {val:.1f}%")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Create model pipeline with SMOTE
        battery_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        
        pipeline = ImbPipeline([
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=self.random_state)),
            ('model', battery_model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nBattery Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Poor/Replace', 'Good', 'Excellent']))
        
        # Save model
        joblib.dump(pipeline, 'battery_condition_classifier.joblib')
        
        # Save feature columns for later use
        joblib.dump(feature_columns, 'battery_feature_columns.joblib')
        
        print("Battery model saved as 'battery_condition_classifier.joblib'")
        
        return pipeline, feature_columns, accuracy
    
    def create_battery_health_functions(self):
        """Create battery health calculation functions"""
        print("\nCreating battery health calculation functions...")
        
        def calculate_battery_health_score(battery_data):
            """Calculate overall battery health score (0-100%)"""
            
            # Voltage health (4.2V is optimal for Li-ion)
            voltage_health = max(0, min(100, (battery_data['voltage'] / 4.2) * 100))
            
            # Capacity health
            capacity_health = battery_data.get('capacity', 80)  # Default 80% if missing
            
            # Temperature impact (25°C is optimal)
            temp_penalty = abs(battery_data['temperature'] - 25) * 2
            temp_health = max(0, 100 - temp_penalty)
            
            # Charge cycle impact
            cycle_health = max(0, 100 - (battery_data.get('charge_cycles', 0) / 20))
            
            # Internal resistance impact (lower is better)
            resistance_health = max(0, 100 - (battery_data['internal_resistance'] * 10000))
            
            # Weighted average
            overall_health = (
                voltage_health * 0.25 +
                capacity_health * 0.30 +
                temp_health * 0.20 +
                cycle_health * 0.15 +
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
        
        return calculate_battery_health_score, predict_battery_remaining_life
    
    def run_complete_analysis(self):
        """Run complete battery analysis and model training"""
        
        # Load and analyze data
        df = self.load_and_analyze_battery_data()
        
        # Engineer features
        df_enhanced = self.engineer_battery_features(df)
        
        # Train model
        model, feature_columns, accuracy = self.train_battery_model(df_enhanced)
        
        # Create health functions
        health_func, life_func = self.create_battery_health_functions()
        
        # Create visualization
        self.create_battery_visualizations(df_enhanced)
        
        print("\n" + "="*80)
        print("BATTERY CONDITION PREDICTOR COMPLETE!")
        print("="*80)
        print(f"Model accuracy: {accuracy:.2%}")
        print("Files generated:")
        print("- battery_condition_classifier.joblib")
        print("- battery_feature_columns.joblib")
        print("- battery_analysis_results.png")
        
        return model, feature_columns, health_func, life_func
    
    def create_battery_visualizations(self, df):
        """Create battery analysis visualizations"""
        print("\nCreating battery visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Battery condition distribution
        condition_counts = df['battery_condition'].value_counts()
        condition_labels = ['Poor/Replace', 'Good', 'Excellent']
        axes[0, 0].pie(condition_counts.values, labels=condition_labels, autopct='%1.1f%%')
        axes[0, 0].set_title('Battery Condition Distribution')
        
        # 2. Voltage vs Health
        scatter = axes[0, 1].scatter(df['voltage'], df['battery_health'], 
                                   c=df['battery_condition'], cmap='RdYlGn', alpha=0.6)
        axes[0, 1].set_xlabel('Voltage (V)')
        axes[0, 1].set_ylabel('Battery Health (%)')
        axes[0, 1].set_title('Voltage vs Battery Health')
        plt.colorbar(scatter, ax=axes[0, 1])
        
        # 3. Charge cycles impact
        axes[0, 2].scatter(df['charge_cycles'], df['battery_health'], alpha=0.6)
        axes[0, 2].set_xlabel('Charge Cycles')
        axes[0, 2].set_ylabel('Battery Health (%)')
        axes[0, 2].set_title('Charge Cycles vs Battery Health')
        
        # 4. Temperature impact
        axes[1, 0].scatter(df['temperature'], df['battery_health'], alpha=0.6)
        axes[1, 0].set_xlabel('Temperature (°C)')
        axes[1, 0].set_ylabel('Battery Health (%)')
        axes[1, 0].set_title('Temperature vs Battery Health')
        
        # 5. Feature correlation heatmap
        feature_cols = ['voltage', 'current', 'temperature', 'capacity', 'internal_resistance']
        corr_matrix = df[feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Battery Feature Correlations')
        
        # 6. Battery health histogram
        axes[1, 2].hist(df['battery_health'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 2].set_xlabel('Battery Health (%)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Battery Health Distribution')
        axes[1, 2].axvline(x=70, color='red', linestyle='--', label='Replacement Threshold')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('battery_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Battery visualizations saved as 'battery_analysis_results.png'")

if __name__ == "__main__":
    # Initialize and run battery condition predictor
    predictor = BatteryConditionPredictor()
    model, features, health_func, life_func = predictor.run_complete_analysis()
    
    print(f"\n🔋 BATTERY CONDITION PREDICTOR READY!")
    print("The system can now predict battery condition and replacement timing.")
    print("Integration with Kafka consumer coming next...")