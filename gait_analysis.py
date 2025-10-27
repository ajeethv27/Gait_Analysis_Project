"""
TENG Gait Analysis System
=========================
Professional ML pipeline for gait classification and bio-authentication

Author: Your Name
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# ===========================================================================
# SECTION 1: DATA LOADING & VALIDATION
# ===========================================================================

def load_and_validate(csv_file='gait_labels.csv'):
    """
    Load labeled data and check completeness
    
    Returns:
        df: DataFrame with validated data
    """
    df = pd.read_csv(csv_file)
    
    print("="*70)
    print("DATA VALIDATION")
    print("="*70)
    print(f"Total recordings: {len(df)}")
    print(f"Subjects: {df['subject_id'].nunique()}")
    print(f"Activities: {df['activity'].unique()}")
    
    # Check missing data
    missing_qom = df['qom'].isna().sum()
    missing_weight = df['weight_kg'].isna().sum()
    missing_height = df['height_cm'].isna().sum()
    
    print(f"\nMissing data:")
    print(f"  QOM ratings: {missing_qom}")
    print(f"  Weight: {missing_weight}")
    print(f"  Height: {missing_height}")
    
    # Get complete records
    complete = df.dropna(subset=['qom', 'weight_kg', 'height_cm'])
    print(f"\nComplete records: {len(complete)} ({len(complete)/len(df)*100:.0f}%)")
    
    if len(complete) > 0:
        print(f"\nQOM Statistics:")
        print(f"  Mean: {complete['qom'].mean():.1f}")
        print(f"  Range: {complete['qom'].min():.0f} - {complete['qom'].max():.0f}")
        print(f"  Std: {complete['qom'].std():.1f}")
    
    print("="*70 + "\n")
    
    return df, complete


def prepare_for_ml(df, output_file='ml_ready.csv'):
    """
    Prepare data for machine learning
    
    Adds derived features:
    - BMI (Body Mass Index)
    - Weight category
    """
    # Add BMI
    df['bmi'] = df['weight_kg'] / (df['height_cm']/100)**2
    
    # Add weight category for classification
    df['weight_category'] = pd.cut(
        df['weight_kg'], 
        bins=[0, 60, 75, 90, 999],
        labels=['light', 'medium', 'heavy', 'very_heavy']
    )
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"✓ ML-ready data saved: {output_file}")
    
    return df


# ===========================================================================
# SECTION 2: WEIGHT PREDICTION (Bio-Authentication)
# ===========================================================================

class WeightPredictor:
    """
    Predict user weight from gait patterns
    
    Use case: Bio-authentication without asking weight
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.feature_names = None
    
    def train(self, features, weights):
        """
        Train weight prediction model
        
        Args:
            features: DataFrame with gait features (stride, amplitude, etc.)
            weights: Actual weights (kg)
        
        Returns:
            mae: Mean absolute error (kg)
            r2: R² score
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, weights, test_size=0.2, random_state=42
        )
        
        self.feature_names = features.columns.tolist()
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print results
        print("="*70)
        print("WEIGHT PREDICTION MODEL (Bio-Authentication)")
        print("="*70)
        print(f"MAE: ±{mae:.1f} kg")
        print(f"R²:  {r2:.3f}")
        
        if mae < 5:
            print("✓ Excellent accuracy!")
        elif mae < 10:
            print("✓ Good accuracy")
        else:
            print("⚠ Need more training data")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop predictive features:")
        for idx, row in importance.head(5).iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.3f}")
        
        print("="*70 + "\n")
        
        return mae, r2, importance
    
    def predict(self, features):
        """Predict weight from gait features"""
        return self.model.predict(features)
    
    def plot_results(self, features, true_weights, save_path='weight_prediction.png'):
        """Visualize prediction accuracy"""
        predictions = self.predict(features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(true_weights, predictions, alpha=0.6, s=100)
        ax1.plot([true_weights.min(), true_weights.max()],
                 [true_weights.min(), true_weights.max()],
                 'r--', linewidth=2, label='Perfect prediction')
        ax1.set_xlabel('Actual Weight (kg)', fontsize=12)
        ax1.set_ylabel('Predicted Weight (kg)', fontsize=12)
        ax1.set_title('Weight Prediction Accuracy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        errors = predictions - true_weights
        ax2.hist(errors, bins=15, edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Prediction Error (kg)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {save_path}")
        plt.close()


# ===========================================================================
# SECTION 3: ACTIVITY CLASSIFICATION
# ===========================================================================

class ActivityClassifier:
    """Classify gait activities: stand, walk, run, jump"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
    
    def train(self, features, activities):
        """
        Train activity classifier
        
        Args:
            features: Gait signal features
            activities: Activity labels (stand/walk/run/jump)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            features, activities, test_size=0.2, stratify=activities, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train)
        test_acc = self.model.score(X_test, y_test)
        
        print("="*70)
        print("ACTIVITY CLASSIFICATION")
        print("="*70)
        print(f"Training accuracy:   {train_acc*100:.1f}%")
        print(f"Testing accuracy:    {test_acc*100:.1f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, self.model.predict(X_test)))
        print("="*70 + "\n")
        
        return train_acc, test_acc
    
    def predict(self, features):
        """Predict activity from features"""
        return self.model.predict(features)


# ===========================================================================
# SECTION 4: QOM ANALYSIS
# ===========================================================================

def analyze_qom_by_demographics(df):
    """
    Analyze how QOM varies with demographics
    
    This helps understand:
    - Do heavier people have lower QOM?
    - Does height affect movement quality?
    """
    complete = df.dropna(subset=['qom', 'weight_kg', 'height_cm', 'bmi'])
    
    if len(complete) < 10:
        print("⚠ Need more complete data for QOM analysis")
        return
    
    print("="*70)
    print("QOM vs DEMOGRAPHICS ANALYSIS")
    print("="*70)
    
    # QOM by weight category
    qom_by_weight = complete.groupby('weight_category')['qom'].agg(['mean', 'std', 'count'])
    print("\nQOM by Weight Category:")
    print(qom_by_weight)
    
    # Correlation analysis
    correlations = complete[['qom', 'weight_kg', 'height_cm', 'bmi']].corr()['qom'].drop('qom')
    print("\nCorrelations with QOM:")
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.3f}")
    
    print("="*70 + "\n")


# ===========================================================================
# MAIN WORKFLOW
# ===========================================================================

def main():
    """Main analysis pipeline"""
    
    print("\n" + "="*70)
    print("TENG GAIT ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Load data
    df_all, df_complete = load_and_validate('gait_labels.csv')
    
    # Step 2: Prepare for ML
    df_ml = prepare_for_ml(df_complete.copy())
    
    # Step 3: QOM analysis
    analyze_qom_by_demographics(df_ml)
    
    print("\n✅ PIPELINE COMPLETE!")
    print("\nNext steps:")
    print("1. Extract features from your CSV files (S01STAND.csv, etc.)")
    print("2. Use WeightPredictor for bio-authentication")
    print("3. Use ActivityClassifier for state detection")
    print("4. Train QOM regression model")
    print("\n" + "="*70)


# ===========================================================================
# USAGE EXAMPLE
# ===========================================================================

if __name__ == "__main__":
    
    # Run main pipeline
    main()
    
    # Example: Weight prediction demo
    print("\n" + "="*70)
    print("DEMO: Weight Prediction from Gait Features")
    print("="*70 + "\n")
    
    # Simulate extracted features (in reality, load from your CSV files)
    np.random.seed(42)
    n_samples = 30
    
    demo_features = pd.DataFrame({
        'peak_amplitude': np.random.uniform(0.3, 0.8, n_samples),
        'stride_duration': np.random.uniform(0.9, 1.4, n_samples),
        'stride_frequency': np.random.uniform(0.7, 1.1, n_samples),
        'signal_energy': np.random.uniform(2.0, 4.0, n_samples),
        'rms_amplitude': np.random.uniform(0.2, 0.6, n_samples),
    })
    
    demo_weights = np.random.uniform(60, 85, n_samples)
    
    # Train and evaluate
    predictor = WeightPredictor()
    mae, r2, importance = predictor.train(demo_features, demo_weights)
    predictor.plot_results(demo_features, demo_weights)
    
    # Test prediction
    new_person = pd.DataFrame({
        'peak_amplitude': [0.55],
        'stride_duration': [1.15],
        'stride_frequency': [0.87],
        'signal_energy': [2.8],
        'rms_amplitude': [0.42],
    })
    
    predicted_weight = predictor.predict(new_person)
    print(f"\n🔍 Predicted weight for new person: {predicted_weight[0]:.1f} kg")
    print("\n" + "="*70)
