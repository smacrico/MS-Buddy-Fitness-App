import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealTimeHealthDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.setup_page_config()
    
    def initialize_session_state(self):
        """Initialize customization settings in session state"""
        if 'metric_preferences' not in st.session_state:
            st.session_state.metric_preferences = {
                'primary_metrics': ['heart_rate', 'steps', 'sleep_efficiency'],
                'update_frequency': 30,  # seconds
                'alert_thresholds': {},
                'visualization_type': 'line',
                'time_range': '24h',
                'color_theme': 'default'
            }
        
        if 'personal_baselines' not in st.session_state:
            st.session_state.personal_baselines = {
                'heart_rate': {'min': 50, 'max': 80, 'target': 65},
                'steps': {'min': 8000, 'max': 12000, 'target': 10000},
                'sleep_efficiency': {'min': 75, 'max': 95, 'target': 85},
                'hrv_score': {'min': 25, 'max': 60, 'target': 40},
                'stress_level': {'min': 10, 'max': 40, 'target': 25},
                'spo2': {'min': 95, 'max': 100, 'target': 98},
                'calories': {'min': 1500, 'max': 3000, 'target': 2200},
                'active_minutes': {'min': 30, 'max': 90, 'target': 60},
                'body_battery': {'min': 50, 'max': 100, 'target': 75},
                'respiration_rate': {'min': 12, 'max': 20, 'target': 16}
            }

    def setup_page_config(self):
        st.set_page_config(
            page_title="Real-Time Health Dashboard",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def get_color_palette(self, theme):
        """Get color palette based on theme selection"""
        color_themes = {
            'default': px.colors.qualitative.Set1,
            'dark': px.colors.qualitative.Dark2,
            'pastel': px.colors.qualitative.Pastel1,
            'health': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            'clinical': ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        }
        return color_themes.get(theme, px.colors.qualitative.Set1)

    def create_customization_sidebar(self):
        """Create comprehensive customization sidebar"""
        st.sidebar.header("🎛️ Dashboard Customization")
        
        # Metric Selection
        st.sidebar.subheader("📊 Metric Selection")
        available_metrics = [
            'heart_rate', 'steps', 'sleep_efficiency', 'hrv_score',
            'stress_level', 'spo2', 'calories', 'active_minutes',
            'body_battery', 'respiration_rate'
        ]
        
        selected_metrics = st.sidebar.multiselect(
            "Choose Primary Metrics",
            available_metrics,
            default=st.session_state.metric_preferences['primary_metrics']
        )
        st.session_state.metric_preferences['primary_metrics'] = selected_metrics
        
        # Visualization Customization
        st.sidebar.subheader("🎨 Visualization Style")
        
        viz_type = st.sidebar.selectbox(
            "Chart Type",
            ["line", "gauge", "area", "bar"],
            index=["line", "gauge", "area", "bar"].index(
                st.session_state.metric_preferences['visualization_type']
            )
        )
        st.session_state.metric_preferences['visualization_type'] = viz_type
        
        # Color Theme
        theme = st.sidebar.selectbox(
            "Color Theme",
            ['default', 'dark', 'pastel', 'health', 'clinical'],
            index=['default', 'dark', 'pastel', 'health', 'clinical'].index(
                st.session_state.metric_preferences['color_theme']
            )
        )
        st.session_state.metric_preferences['color_theme'] = theme
        
        # Time Range Selection
        st.sidebar.subheader("⏰ Time Range")
        time_range = st.sidebar.selectbox(
            "Display Period",
            ["1h", "6h", "12h", "24h", "7d", "30d"],
            index=["1h", "6h", "12h", "24h", "7d", "30d"].index(
                st.session_state.metric_preferences['time_range']
            )
        )
        st.session_state.metric_preferences['time_range'] = time_range
        
        # Update Frequency
        update_freq = st.sidebar.slider(
            "Update Frequency (seconds)",
            min_value=5,
            max_value=300,
            value=st.session_state.metric_preferences['update_frequency'],
            step=5
        )
        st.session_state.metric_preferences['update_frequency'] = update_freq
        
        return selected_metrics, viz_type, theme, time_range, update_freq

    def create_baseline_configuration(self):
        """Allow users to set personal baselines and alert thresholds"""
        st.sidebar.subheader("🎯 Personal Baselines")
        
        # Expandable sections for each metric
        for metric in st.session_state.metric_preferences['primary_metrics']:
            with st.sidebar.expander(f"{metric.replace('_', ' ').title()} Settings"):
                if metric not in st.session_state.personal_baselines:
                    st.session_state.personal_baselines[metric] = {
                        'min': 0, 'max': 100, 'target': 50
                    }
                
                baseline = st.session_state.personal_baselines[metric]
                
                # Custom range settings for different metrics
                metric_configs = {
                    'heart_rate': {'min': 40, 'max': 200, 'default': 65},
                    'steps': {'min': 0, 'max': 20000, 'default': 10000},
                    'sleep_efficiency': {'min': 50, 'max': 100, 'default': 85},
                    'hrv_score': {'min': 10, 'max': 80, 'default': 40},
                    'stress_level': {'min': 0, 'max': 100, 'default': 25},
                    'spo2': {'min': 85, 'max': 100, 'default': 98},
                    'calories': {'min': 1000, 'max': 5000, 'default': 2200},
                    'active_minutes': {'min': 0, 'max': 180, 'default': 60},
                    'body_battery': {'min': 0, 'max': 100, 'default': 75},
                    'respiration_rate': {'min': 8, 'max': 30, 'default': 16}
                }
                
                config = metric_configs.get(metric, {'min': 0, 'max': 100, 'default': 50})
                
                # Target setting
                target = st.slider(
                    f"Target {metric.replace('_', ' ').title()}",
                    min_value=config['min'],
                    max_value=config['max'],
                    value=baseline.get('target', config['default']),
                    key=f"target_{metric}"
                )
                
                # Alert thresholds
                col1, col2 = st.columns(2)
                with col1:
                    low_threshold = st.number_input(
                        "Low Alert",
                        min_value=config['min'],
                        max_value=config['max'],
                        value=baseline.get('min', config['min']),
                        key=f"low_{metric}"
                    )
                
                with col2:
                    high_threshold = st.number_input(
                        "High Alert",
                        min_value=config['min'],
                        max_value=config['max'],
                        value=baseline.get('max', config['max']),
                        key=f"high_{metric}"
                    )
                
                # Update session state
                st.session_state.personal_baselines[metric] = {
                    'min': low_threshold,
                    'max': high_threshold,
                    'target': target
                }

    def create_real_time_chart(self, metric_name, data, chart_type, color_theme):
        """Generate customizable real-time charts"""
        colors = self.get_color_palette(color_theme)
        baseline = st.session_state.personal_baselines.get(metric_name, {})
        
        if chart_type == "gauge":
            return self.create_gauge_chart(metric_name, data, colors, baseline)
        elif chart_type == "area":
            return self.create_area_chart(metric_name, data, colors, baseline)
        elif chart_type == "bar":
            return self.create_bar_chart(metric_name, data, colors, baseline)
        else:
            return self.create_line_chart(metric_name, data, colors, baseline)

    def create_gauge_chart(self, metric_name, data, colors, baseline):
        """Create customizable gauge chart for real-time metrics"""
        current_value = data.iloc[-1] if len(data) > 0 else 0
        target = baseline.get('target', 50)
        min_val = baseline.get('min', 0)
        max_val = baseline.get('max', 100)
        
        # Determine gauge range
        gauge_max = max_val * 1.2  # 20% buffer above max threshold
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = current_value,
            delta = {'reference': target, 'position': "top"},
            title = {'text': f"{metric_name.replace('_', ' ').title()}"},
            gauge = {
                'axis': {'range': [None, gauge_max]},
                'bar': {'color': colors[0]},
                'steps': [
                    {'range': [0, min_val], 'color': "lightcoral"},
                    {'range': [min_val, target], 'color': "lightgreen"},
                    {'range': [target, max_val], 'color': "gold"},
                    {'range': [max_val, gauge_max], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        return fig

    def create_line_chart(self, metric_name, data, colors, baseline):
        """Create customizable line chart with baseline zones"""
        fig = go.Figure()
        
        # Add baseline zones
        if baseline:
            fig.add_hrect(
                y0=baseline.get('min', 0), 
                y1=baseline.get('target', 50),
                fillcolor="lightgreen", 
                opacity=0.2,
                line_width=0,
                annotation_text="Good Zone"
            )
            
            fig.add_hline(
                y=baseline.get('target', 50),
                line_dash="dash",
                line_color=colors[0],
                annotation_text="Target"
            )
        
        # Add data line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines+markers',
            name=metric_name.replace('_', ' ').title(),
            line=dict(color=colors[0], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f"Real-Time {metric_name.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=400,
            showlegend=True
        )
        
        return fig

    def create_area_chart(self, metric_name, data, colors, baseline):
        """Create area chart with baseline zones"""
        fig = go.Figure()
        
        # Add baseline zones
        if baseline:
            fig.add_hrect(
                y0=baseline.get('min', 0), 
                y1=baseline.get('target', 50),
                fillcolor="lightgreen", 
                opacity=0.1,
                line_width=0
            )
        
        # Add area chart
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            fill='tozeroy',
            name=metric_name.replace('_', ' ').title(),
            line=dict(color=colors[0], width=2),
            fillcolor=f"rgba{tuple(list(px.colors.hex_to_rgb(colors[0])) + [0.3])}"
        ))
        
        fig.update_layout(
            title=f"Real-Time {metric_name.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=400
        )
        
        return fig

    def create_bar_chart(self, metric_name, data, colors, baseline):
        """Create bar chart for discrete time periods"""
        # Resample data to hourly for bar chart
        hourly_data = data.resample('1H').mean()
        
        fig = go.Figure()
        
        # Add target line
        if baseline:
            fig.add_hline(
                y=baseline.get('target', 50),
                line_dash="dash",
                line_color="red",
                annotation_text="Target"
            )
        
        # Add bars
        fig.add_trace(go.Bar(
            x=hourly_data.index,
            y=hourly_data.values,
            name=metric_name.replace('_', ' ').title(),
            marker_color=colors[0]
        ))
        
        fig.update_layout(
            title=f"Hourly {metric_name.replace('_', ' ').title()}",
            xaxis_title="Time",
            yaxis_title=metric_name.replace('_', ' ').title(),
            height=400
        )
        
        return fig

    def create_adaptive_layout(self):
        """Create adaptive dashboard layout based on user preferences"""
        
        # Header with real-time status
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.title("🏥 Real-Time Health Dashboard")
        
        with col2:
            st.metric("Status", "🟢 Live", help="Dashboard is running in real-time mode")
        
        with col3:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric("Last Update", current_time)
        
        # Dynamic metric grid
        metrics = st.session_state.metric_preferences['primary_metrics']
        chart_type = st.session_state.metric_preferences['visualization_type']
        
        if not metrics:
            st.warning("Please select at least one metric from the sidebar to display.")
            return
        
        # Responsive grid layout
        if len(metrics) == 1:
            cols = st.columns(1)
        elif len(metrics) == 2:
            cols = st.columns(2)
        elif len(metrics) <= 4:
            cols = st.columns(2)
        else:
            cols = st.columns(3)
        
        # Generate charts for each metric
        for i, metric in enumerate(metrics):
            col_idx = i % len(cols)
            
            with cols[col_idx]:
                # Generate sample real-time data
                data = self.generate_real_time_data(metric)
                
                # Create chart
                fig = self.create_real_time_chart(
                    metric, 
                    data, 
                    chart_type, 
                    st.session_state.metric_preferences['color_theme']
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add metric summary
                current_value = data.iloc[-1] if len(data) > 0 else 0
                baseline = st.session_state.personal_baselines.get(metric, {})
                target = baseline.get('target', 'N/A')
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Current", f"{current_value:.1f}")
                with col_b:
                    st.metric("Target", f"{target}")
                with col_c:
                    # Calculate change from previous value
                    if len(data) > 1:
                        change = ((current_value - data.iloc[-2]) / data.iloc[-2]) * 100
                        st.metric("Change", f"{change:+.1f}%")
                    else:
                        st.metric("Change", "N/A")

    def generate_real_time_data(self, metric):
        """Generate sample real-time data for demonstration"""
        time_range = st.session_state.metric_preferences['time_range']
        
        # Convert time range to hours
        hours = {
            '1h': 1, '6h': 6, '12h': 12, '24h': 24, '7d': 168, '30d': 720
        }[time_range]
        
        # Generate timestamps
        now = datetime.now()
        timestamps = pd.date_range(
            start=now - timedelta(hours=hours),
            end=now,
            freq='5T'  # 5-minute intervals
        )
        
        # Generate realistic data based on metric type
        baseline = st.session_state.personal_baselines.get(metric, {})
        target = baseline.get('target', 50)
        
        # Add some realistic variation based on metric type
        if metric == 'heart_rate':
            values = np.random.normal(target, 8, len(timestamps))
            # Add some daily rhythm
            hours_array = np.array([t.hour for t in timestamps])
            daily_variation = 10 * np.sin(2 * np.pi * hours_array / 24)
            values += daily_variation
        elif metric == 'steps':
            # Steps accumulate throughout the day
            hourly_steps = np.random.poisson(target/24, len(timestamps))
            values = np.cumsum(hourly_steps)
        elif metric == 'sleep_efficiency':
            # Sleep efficiency varies less
            values = np.random.normal(target, 5, len(timestamps))
        elif metric == 'body_battery':
            # Body battery depletes during day, recharges at night
            hours_array = np.array([t.hour for t in timestamps])
            daily_pattern = 30 * np.cos(2 * np.pi * (hours_array - 6) / 24)
            values = target + daily_pattern + np.random.normal(0, 5, len(timestamps))
        else:
            values = np.random.normal(target, target * 0.15, len(timestamps))
        
        # Ensure values stay within reasonable bounds
        min_val = baseline.get('min', 0) * 0.8
        max_val = baseline.get('max', 100) * 1.2
        values = np.clip(values, min_val, max_val)
        
        return pd.Series(values, index=timestamps)

    def create_alert_system(self):
        """Create customizable real-time alert system"""
        
        st.subheader("🚨 Real-Time Health Alerts")
        
        # Alert configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            alert_container = st.container()
            
        with col2:
            st.subheader("Alert Settings")
            
            # Alert types
            alert_types = st.multiselect(
                "Enable Alerts For:",
                ["Threshold Breaches", "Trend Changes", "Anomalies", "Targets Reached"],
                default=["Threshold Breaches", "Targets Reached"]
            )
            
            # Alert sensitivity
            sensitivity = st.slider(
                "Alert Sensitivity",
                min_value=1,
                max_value=10,
                value=5,
                help="Higher values = more sensitive to changes"
            )
        
        # Generate alerts based on current data
        alerts = self.check_health_alerts(alert_types, sensitivity)
        
        with alert_container:
            if alerts:
                for alert in alerts:
                    alert_color = "🔴" if alert['severity'] == 'high' else "🟡" if alert['severity'] == 'medium' else "🟢"
                    
                    if alert['severity'] == 'high':
                        st.error(f"{alert_color} **{alert['metric']}**: {alert['message']} (Value: {alert['value']})")
                    elif alert['severity'] == 'medium':
                        st.warning(f"{alert_color} **{alert['metric']}**: {alert['message']} (Value: {alert['value']})")
                    else:
                        st.success(f"{alert_color} **{alert['metric']}**: {alert['message']} (Value: {alert['value']})")
            else:
                st.info("✅ All metrics are within normal ranges")

    def check_health_alerts(self, alert_types, sensitivity):
        """Check for health metric alerts"""
        alerts = []
        current_time = datetime.now()
        
        for metric in st.session_state.metric_preferences['primary_metrics']:
            baseline = st.session_state.personal_baselines.get(metric, {})
            
            # Generate current value
            data = self.generate_real_time_data(metric)
            current_value = data.iloc[-1] if len(data) > 0 else baseline.get('target', 50)
            
            # Check threshold breaches
            if "Threshold Breaches" in alert_types:
                if current_value < baseline.get('min', 0):
                    alerts.append({
                        'metric': metric.replace('_', ' ').title(),
                        'message': f"Below minimum threshold ({baseline.get('min', 0)})",
                        'value': round(current_value, 1),
                        'severity': 'high',
                        'timestamp': current_time
                    })
                elif current_value > baseline.get('max', 100):
                    alerts.append({
                        'metric': metric.replace('_', ' ').title(),
                        'message': f"Above maximum threshold ({baseline.get('max', 100)})",
                        'value': round(current_value, 1),
                        'severity': 'high',
                        'timestamp': current_time
                    })
            
            # Check target achievement
            if "Targets Reached" in alert_types:
                target = baseline.get('target', 50)
                tolerance = target * 0.05  # 5% tolerance
                if abs(current_value - target) <= tolerance:
                    alerts.append({
                        'metric': metric.replace('_', ' ').title(),
                        'message': f"Target achieved! 🎯",
                        'value': round(current_value, 1),
                        'severity': 'low',
                        'timestamp': current_time
                    })
        
        return alerts

    def create_custom_metric_builder(self):
        """Allow users to create custom composite health metrics"""
        
        st.subheader("🔧 Custom Metric Builder")
        
        with st.expander("Create Custom Health Metric"):
            metric_name = st.text_input(
                "Custom Metric Name", 
                placeholder="e.g., My Wellness Score"
            )
            
            st.write("**Formula Builder:**")
            
            # Available base metrics
            base_metrics = st.session_state.metric_preferences['primary_metrics']
            
            if not base_metrics:
                st.warning("Please select some primary metrics first to create custom formulas.")
                return
            
            # Formula components
            formula_components = []
            
            st.write("Build your custom metric using the selected primary metrics:")
            
            for i in range(min(3, len(base_metrics))):  # Allow up to 3 components or number of available metrics
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    component_metric = st.selectbox(
                        f"Component {i+1}",
                        ["None"] + base_metrics,
                        key=f"component_{i}"
                    )
                
                with col2:
                    weight = st.number_input(
                        "Weight",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.33,
                        step=0.01,
                        key=f"weight_{i}"
                    )
                
                with col3:
                    if i < 2:  # Don't show operation for the last component
                        operation = st.selectbox(
                            "Operation",
                            ["+", "-", "*", "/"],
                            key=f"operation_{i}"
                        )
                    else:
                        operation = ""
                
                if component_metric != "None":
                    formula_components.append({
                        'metric': component_metric,
                        'weight': weight,
                        'operation': operation if i < len(base_metrics) - 1 else ""
                    })
            
            # Display formula
            if formula_components and metric_name:
                formula_parts = []
                for i, comp in enumerate(formula_components):
                    part = f"({comp['weight']} * {comp['metric']})"
                    if i < len(formula_components) - 1 and comp['operation']:
                        part += f" {comp['operation']} "
                    formula_parts.append(part)
                
                formula_text = "".join(formula_parts)
                st.code(f"{metric_name} = {formula_text}")
                
                if st.button("Create Custom Metric"):
                    # Store custom metric in session state
                    if 'custom_metrics' not in st.session_state:
                        st.session_state.custom_metrics = {}
                    
                    st.session_state.custom_metrics[metric_name] = {
                        'components': formula_components,
                        'formula': formula_text
                    }
                    st.success(f"✅ Custom metric '{metric_name}' created!")
            
            # Display existing custom metrics
            if 'custom_metrics' in st.session_state and st.session_state.custom_metrics:
                st.write("**Your Custom Metrics:**")
                for name, details in st.session_state.custom_metrics.items():
                    with st.expander(f"📊 {name}"):
                        st.code(details['formula'])
                        if st.button(f"Delete {name}", key=f"delete_{name}"):
                            del st.session_state.custom_metrics[name]
                            st.rerun()
