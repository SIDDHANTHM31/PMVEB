import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# Load your current test data
df_test = pd.read_csv('engines_dataset-test.csv')

# Remove duplicates first
df_test_clean = df_test.drop_duplicates()
print(f"Removed {len(df_test) - len(df_test_clean)} duplicate rows")

def synthesize_engine_data(df, n_samples=1000, method='gaussian_mixture'):
    """
    Synthesize realistic engine sensor data
    """
    # Separate features and target
    feature_cols = ['engineRpm', 'lubOilPressure', 'fuelPressure', 
                   'coolantPressure', 'lubOilTemp', 'coolantTemp']
    X = df[feature_cols].values
    y = df['engineCondition'].values
    
    synthesized_data = []
    
    for condition in np.unique(y):
        # Get data for this condition
        condition_data = X[y == condition]
        n_condition_samples = int(n_samples * np.sum(y == condition) / len(y))
        
        if method == 'gaussian_mixture':
            # Fit Gaussian Mixture Model
            gmm = GaussianMixture(n_components=min(3, len(condition_data)//10), 
                                random_state=42)
            gmm.fit(condition_data)
            
            # Generate synthetic samples
            synthetic_samples = gmm.sample(n_condition_samples)[0]
            
        elif method == 'multivariate_normal':
            # Fit multivariate normal distribution
            mean = np.mean(condition_data, axis=0)
            cov = np.cov(condition_data.T)
            
            # Add small regularization to ensure positive definite
            cov += np.eye(cov.shape[0]) * 1e-6
            
            # Generate synthetic samples
            synthetic_samples = multivariate_normal.rvs(
                mean=mean, cov=cov, size=n_condition_samples
            )
            
        elif method == 'noise_injection':
            # Add controlled noise to existing samples
            n_repeats = n_condition_samples // len(condition_data) + 1
            base_samples = np.tile(condition_data, (n_repeats, 1))[:n_condition_samples]
            
            # Add noise (5% of standard deviation)
            noise_std = np.std(condition_data, axis=0) * 0.05
            noise = np.random.normal(0, noise_std, base_samples.shape)
            synthetic_samples = base_samples + noise
        
        # Apply realistic constraints
        synthetic_samples = apply_engine_constraints(synthetic_samples, feature_cols)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_cols)
        synthetic_df['engineCondition'] = condition
        synthesized_data.append(synthetic_df)
    
    return pd.concat(synthesized_data, ignore_index=True)

def apply_engine_constraints(data, feature_cols):
    """
    Apply realistic physical constraints to engine data
    """
    data = data.copy()
    
    # Define realistic ranges for each parameter
    constraints = {
        'engineRpm': (300, 2000),      # RPM range
        'lubOilPressure': (1.0, 6.0),  # Oil pressure range
        'fuelPressure': (1.0, 20.0),   # Fuel pressure range
        'coolantPressure': (0.5, 7.0), # Coolant pressure range
        'lubOilTemp': (70, 95),         # Oil temperature range
        'coolantTemp': (65, 100)        # Coolant temperature range
    }
    
    for i, col in enumerate(feature_cols):
        if col in constraints:
            min_val, max_val = constraints[col]
            data[:, i] = np.clip(data[:, i], min_val, max_val)
    
    return data

def create_enhanced_test_set(original_df, target_size=2000):
    """
    Create enhanced test set with original + synthetic data
    """
    # Clean original data
    original_clean = original_df.drop_duplicates()
    
    # Calculate how much synthetic data to generate
    synthetic_size = max(0, target_size - len(original_clean))
    
    if synthetic_size > 0:
        # Generate synthetic data using multiple methods
        synthetic_gmm = synthesize_engine_data(original_clean, 
                                             synthetic_size//2, 
                                             'gaussian_mixture')
        synthetic_noise = synthesize_engine_data(original_clean, 
                                               synthetic_size//2, 
                                               'noise_injection')
        
        # Combine all data
        enhanced_df = pd.concat([
            original_clean,
            synthetic_gmm,
            synthetic_noise
        ], ignore_index=True)
    else:
        enhanced_df = original_clean
    
    # Shuffle the data
    enhanced_df = enhanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return enhanced_df

# Generate enhanced test set
enhanced_test_data = create_enhanced_test_set(df_test, target_size=2000)

# Save enhanced test set
enhanced_test_data.to_csv('engines_dataset-test_enhanced.csv', index=False)

# Print statistics
print("\nDataset Statistics:")
print(f"Original test set: {len(df_test)} samples")
print(f"Clean test set: {len(df_test.drop_duplicates())} samples") 
print(f"Enhanced test set: {len(enhanced_test_data)} samples")

print("\nClass distribution in enhanced test set:")
print(enhanced_test_data['engineCondition'].value_counts().sort_index())

# Quality checks
def quality_check(df, name):
    print(f"\n{name} Quality Check:")
    print(f"- Missing values: {df.isnull().sum().sum()}")
    print(f"- Duplicate rows: {df.duplicated().sum()}")
    print(f"- Class balance: {df['engineCondition'].value_counts(normalize=True).round(3).to_dict()}")
    
    # Check feature ranges
    feature_cols = ['engineRpm', 'lubOilPressure', 'fuelPressure', 
                   'coolantPressure', 'lubOilTemp', 'coolantTemp']
    print(f"- Feature ranges:")
    for col in feature_cols:
        print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

quality_check(df_test, "Original")
quality_check(enhanced_test_data, "Enhanced")