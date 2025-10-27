"""
Generate Sample Signal Visualization
=====================================
Create publication-quality plots showing TENG signals for different activities

This generates: plot_sample_signals.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def plot_sample_signals(data_folder='User_Data_Labelled', output_file='plot_sample_signals.png'):
    """
    Plot sample signals for each activity type
    
    Args:
        data_folder: Path to folder with oscilloscope CSVs
        output_file: Output filename for plot
    """
    
    print("="*70)
    print("GENERATING SAMPLE SIGNAL PLOTS")
    print("="*70)
    
    # Find sample files for each activity
    activities = ['stand', 'walk', 'run', 'jump']
    sample_files = {}
    
    for activity in activities:
        # Try to find S01's file for this activity
        pattern = f"S01{activity.upper()}*.csv"
        files = glob.glob(str(Path(data_folder) / pattern))
        
        if files:
            sample_files[activity] = files[0]
            print(f"✓ Found {activity}: {Path(files[0]).name}")
        else:
            print(f"⚠ No file found for {activity}")
    
    if not sample_files:
        print("❌ No data files found. Check data_folder path.")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'stand': '#2ecc71', 'walk': '#3498db', 'run': '#e74c3c', 'jump': '#f39c12'}
    
    for idx, activity in enumerate(activities):
        ax = axes[idx]
        
        if activity not in sample_files:
            ax.text(0.5, 0.5, f'No data for {activity}', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            ax.set_title(f'{activity.capitalize()} (No Data)', fontsize=14, fontweight='bold')
            continue
        
        # Load and plot data
        try:
            df = pd.read_csv(sample_files[activity])
            
            # Handle different possible column names
            if 'Time(s)' in df.columns and 'Voltage(V)' in df.columns:
                time = df['Time(s)'].values
                voltage = df['Voltage(V)'].values
            elif 'time' in df.columns and 'voltage' in df.columns:
                time = df['time'].values
                voltage = df['voltage'].values
            else:
                # Assume first two columns
                time = df.iloc[:, 0].values
                voltage = df.iloc[:, 1].values
            
            # Plot only first 10 seconds for clarity
            mask = time <= 10
            time_plot = time[mask]
            voltage_plot = voltage[mask]
            
            ax.plot(time_plot, voltage_plot, 
                   color=colors[activity], linewidth=1.5, alpha=0.8)
            
            # Add statistics
            mean_v = np.mean(voltage_plot)
            std_v = np.std(voltage_plot)
            max_v = np.max(voltage_plot)
            
            # Add text box with stats
            stats_text = f'Max: {max_v:.2f}V\nMean: {mean_v:.2f}V\nStd: {std_v:.2f}V'
            ax.text(0.98, 0.97, stats_text, 
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
            
            # Styling
            ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Voltage (V)', fontsize=11, fontweight='bold')
            ax.set_title(f'{activity.capitalize()}', 
                        fontsize=14, fontweight='bold', color=colors[activity])
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 10])
            
            # Add activity-specific annotations
            if activity == 'stand':
                ax.text(5, ax.get_ylim()[1]*0.5, 
                       'Minimal movement\nNear-zero signal', 
                       ha='center', fontsize=10, style='italic', 
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            elif activity == 'walk':
                # Find peaks (stride events)
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(voltage_plot, height=np.max(voltage_plot)*0.3, distance=50)
                if len(peaks) > 0:
                    ax.plot(time_plot[peaks], voltage_plot[peaks], 'ro', markersize=8, 
                           label=f'Stride events (n={len(peaks)})')
                    ax.legend(loc='upper left', fontsize=9)
            
            elif activity == 'run':
                ax.text(5, ax.get_ylim()[1]*0.8, 
                       'Higher amplitude\nFaster frequency', 
                       ha='center', fontsize=10, style='italic',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            elif activity == 'jump':
                # Find jump events
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(voltage_plot, height=np.max(voltage_plot)*0.5, distance=30)
                if len(peaks) > 0:
                    ax.plot(time_plot[peaks], voltage_plot[peaks], 'r^', markersize=10, 
                           label=f'Jump events (n={len(peaks)})')
                    ax.legend(loc='upper left', fontsize=9)
            
            print(f"  ✓ Plotted {activity}: {len(time_plot)} samples, max={max_v:.2f}V")
            
        except Exception as e:
            print(f"  ✗ Error plotting {activity}: {str(e)}")
            ax.text(0.5, 0.5, f'Error loading {activity}', 
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
    
    # Overall title
    fig.suptitle('TENG Gait Sensor: Sample Signals for Different Activities', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # Add footer with info
    fig.text(0.5, 0.01, 
            'Dataset: 12 subjects, 84 recordings | Sensor: TENG (Copper/PTFE) | Sampling: ~100-200 Hz',
            ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.99])
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot: {output_file}")
    plt.close()
    
    print("="*70 + "\n")


def plot_signal_comparison(data_folder='User_Data_Labelled', 
                          output_file='signal_comparison_all_activities.png'):
    """
    Plot all 4 activities in one figure for direct comparison
    """
    
    print("Generating comparison plot...")
    
    activities = ['stand', 'walk', 'run', 'jump']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'stand': '#2ecc71', 'walk': '#3498db', 'run': '#e74c3c', 'jump': '#f39c12'}
    offset = 0
    offset_step = 2  # Vertical separation between activities
    
    for activity in activities:
        pattern = f"S01{activity.upper()}*.csv"
        files = glob.glob(str(Path(data_folder) / pattern))
        
        if not files:
            continue
        
        try:
            df = pd.read_csv(files[0])
            
            if 'Time(s)' in df.columns and 'Voltage(V)' in df.columns:
                time = df['Time(s)'].values
                voltage = df['Voltage(V)'].values
            else:
                time = df.iloc[:, 0].values
                voltage = df.iloc[:, 1].values
            
            # Plot first 10 seconds
            mask = time <= 10
            time_plot = time[mask]
            voltage_plot = voltage[mask] + offset
            
            ax.plot(time_plot, voltage_plot, 
                   color=colors[activity], linewidth=1.2, 
                   label=f'{activity.capitalize()}', alpha=0.9)
            
            offset += offset_step
            
        except Exception as e:
            print(f"Could not plot {activity}: {e}")
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Voltage (V) [Offset for visibility]', fontsize=12, fontweight='bold')
    ax.set_title('TENG Signal Comparison: All Activities', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 10])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}\n")
    plt.close()


if __name__ == "__main__":
    
    # Generate both plots
    plot_sample_signals()
    plot_signal_comparison()
    
    print("✅ Visualization complete!")
    print("\nGenerated files:")
    print("  • plot_sample_signals.png")
    print("  • signal_comparison_all_activities.png")
    print("\nAdd these to your thesis or dataset package!")
