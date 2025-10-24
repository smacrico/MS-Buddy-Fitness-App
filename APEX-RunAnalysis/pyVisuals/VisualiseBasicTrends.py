import pandas as pd
import matplotlib.pyplot as plt

def visualize_basic_trends(dataset):
    """
    Power BI compatible visualization for running trends
    dataset: Power BI DataFrame input
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Convert date to datetime if needed
        dataset['date'] = pd.to_datetime(dataset['date'])
        
        # Plot 1: Running Economy over time
        plt.subplot(2, 2, 1)
        plt.plot(dataset['date'], dataset['running_economy'], 'b-o')
        plt.title('Running Economy Trend')
        plt.xticks(rotation=45)
        plt.ylabel('Running Economy')
        
        # Plot 2: Efficiency Score over time
        plt.subplot(2, 2, 2)
        plt.plot(dataset['date'], dataset['efficiency_score'], 'g-o')
        plt.title('Efficiency Score Trend')
        plt.xticks(rotation=45)
        plt.ylabel('Efficiency Score')
        
        # Plot 3: Energy Cost vs Distance
        plt.subplot(2, 2, 3)
        plt.scatter(dataset['distance'], dataset['energy_cost'])
        plt.title('Energy Cost vs Distance')
        plt.xlabel('Distance (km)')
        plt.ylabel('Energy Cost')
        
        # Plot 4: Heart Rate vs Running Economy
        plt.subplot(2, 2, 4)
        plt.scatter(dataset['heart_rate'], dataset['running_economy'])
        plt.title('Heart Rate vs Running Economy')
        plt.xlabel('Heart Rate (bpm)')
        plt.ylabel('Running Economy')
        
        plt.tight_layout()
        
    except Exception as e:
        plt.figure()
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        
    return plt