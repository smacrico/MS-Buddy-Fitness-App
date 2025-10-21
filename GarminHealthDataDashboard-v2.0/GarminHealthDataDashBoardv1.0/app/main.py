import streamlit as st
import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from dashboard import RealTimeHealthDashboard
import time

def main():
    """Main application entry point"""
    
    # Initialize dashboard
    dashboard = RealTimeHealthDashboard()
    
    # Create sidebar customization
    selected_metrics, viz_type, theme, time_range, update_freq = dashboard.create_customization_sidebar()
    dashboard.create_baseline_configuration()
    
    # Main dashboard layout
    dashboard.create_adaptive_layout()
    
    # Alert system
    dashboard.create_alert_system()
    
    # Custom metrics builder
    dashboard.create_custom_metric_builder()
    
    # Auto-refresh functionality
    if st.sidebar.checkbox("🔄 Enable Auto-Refresh"):
        # Create a placeholder for the refresh timer
        refresh_placeholder = st.sidebar.empty()
        
        # Countdown timer
        for i in range(update_freq, 0, -1):
            refresh_placeholder.text(f"Next refresh in: {i}s")
            time.sleep(1)
        
        refresh_placeholder.empty()
        st.rerun()

if __name__ == "__main__":
    main()
