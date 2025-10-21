import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from db_helper import HealthDataManager

class RealTimeHealthDashboard:
    def __init__(self):
        self.db_manager = HealthDataManager()
        self.user_id = 1  # Could add login/user selection if needed
        self.initialize_session_state()
        self.setup_page_config()
        self.load_user_baselines()

    def initialize_session_state(self):
        """Initialize user dashboard preferences."""
        if 'metric_preferences' not in st.session_state:
            st.session_state.metric_preferences = {
                'primary_metrics': ['heart_rate', 'steps', 'sleep_efficiency'],
                'update_frequency': 30,
                'visualization_type': 'line',
                'time_range': '24h',
                'color_theme': 'default'
            }

    def setup_page_config(self):
        st.set_page_config(
            page_title="Real-Time Health Dashboard",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def load_user_baselines(self):
        """Load user baseline settings from the database."""
        baselines = self.db_manager.get_user_baselines(self.user_id)
        st.session_state.personal_baselines = baselines

    def create_customization_sidebar(self):
        """Build sidebar with dashboard controls."""
        st.sidebar.header("🎛️ Dashboard Customization")
        available_metrics = self.db_manager.get_available_metrics()
        selected_metrics = st.sidebar.multiselect(
            "Choose Primary Metrics",
            available_metrics,
            default=st.session_state.metric_preferences.get('primary_metrics', available_metrics[:3])
        )
        st.session_state.metric_preferences['primary_metrics'] = selected_metrics

        viz_type = st.sidebar.selectbox(
            "Chart Type",
            ["line", "gauge", "area", "bar"],
            index=["line", "gauge", "area", "bar"].index(
                st.session_state.metric_preferences['visualization_type']
            )
        )
        st.session_state.metric_preferences['visualization_type'] = viz_type

        theme = st.sidebar.selectbox(
            "Color Theme",
            ['default', 'dark', 'pastel', 'health', 'clinical'],
            index=['default', 'dark', 'pastel', 'health', 'clinical'].index(
                st.session_state.metric_preferences['color_theme']
            )
        )
        st.session_state.metric_preferences['color_theme'] = theme

        time_range = st.sidebar.selectbox(
            "Display Period",
            ["1h", "6h", "12h", "24h", "7d", "30d"],
            index=["1h", "6h", "12h", "24h", "7d", "30d"].index(
                st.session_state.metric_preferences['time_range']
            )
        )
        st.session_state.metric_preferences['time_range'] = time_range

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
        """Let users adjust baselines (persist changes to DB as enhancement)."""
        st.sidebar.subheader("🎯 Personal Baselines")
        # For now: display DB-loaded baselines; editing can be added if you desire
        for metric in st.session_state.metric_preferences['primary_metrics']:
            b = st.session_state.personal_baselines.get(metric, {})
            with st.sidebar.expander(f"{metric.replace('_', ' ').title()} Baseline"):
                st.write(f"**Target:** {b.get('target', 'N/A')}")
                st.write(f"**Min Threshold:** {b.get('min', 'N/A')}")
                st.write(f"**Max Threshold:** {b.get('max', 'N/A')}")

    def get_color_palette(self, theme):
        color_themes = {
            'default': px.colors.qualitative.Set1,
            'dark': px.colors.qualitative.Dark2,
            'pastel': px.colors.qualitative.Pastel1,
            'health': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
            'clinical': ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
        }
        return color_themes.get(theme, px.colors.qualitative.Set1)

    def load_data_from_db(self, metric):
        """Load real data for the metric from database."""
        time_range = st.session_state.metric_preferences['time_range']
        hours = {'1h': 1, '6h': 6, '12h': 12, '24h': 24, '7d': 168, '30d': 720}[time_range]
        s = self.db_manager.get_metric_data(metric, hours, self.user_id)
        # If data empty, fallback to a dummy Series (to avoid breaking plots)
        if s.empty:
            now = datetime.now()
            idx = pd.date_range(now - timedelta(hours=hours), now, freq='H')
            s = pd.Series([np.nan]*len(idx), index=idx)
        return s

    def create_real_time_chart(self, metric_name, data, chart_type, color_theme):
        colors = self.get_color_palette(color_theme)
        baseline = st.session_state.personal_baselines.get(metric_name, {})
        # The chart plotting code (line, gauge, area, or bar) can be reused from earlier examples
        # For brevity, I'll provide just the line chart as core template; extend with gauges, bars if desired
        if chart_type == "gauge":
            current_value = data.dropna().iloc[-1] if len(data.dropna()) > 0 else 0
            target = baseline.get('target', 50)
            min_val = baseline.get('min', 0)
            max_val = baseline.get('max', 100)
            gauge_max = max_val * 1.2
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_value,
                delta={'reference': target, 'position': "top"},
                title={'text': f"{metric_name.replace('_', ' ').title()}"},
                gauge={
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
            fig.update_layout(height=300)
            return fig

        else:  # default to line chart
            fig = go.Figure()
            if baseline:
                fig.add_hrect(y0=baseline.get('min', 0), y1=baseline.get('target', 50),
                              fillcolor="lightgreen", opacity=0.2, line_width=0)
                fig.add_hline(y=baseline.get('target', 50), line_dash="dash",
                              line_color=colors[0], annotation_text="Target")
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
                showlegend=False
            )
            return fig

    def create_adaptive_layout(self):
        """Main dashboard metrics and plots."""
        metrics = st.session_state.metric_preferences['primary_metrics']
        chart_type = st.session_state.metric_preferences['visualization_type']

        if not metrics:
            st.warning("Please select at least one metric from the sidebar to display.")
            return

        st.title("🏥 Real-Time Health Dashboard")
        for i, metric in enumerate(metrics):
            data = self.load_data_from_db(metric)
            col1, col2 = st.columns([3,1])
            with col1:
                fig = self.create_real_time_chart(
                    metric, data, chart_type,
                    st.session_state.metric_preferences['color_theme']
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                current_value = data.dropna().iloc[-1] if len(data.dropna()) > 0 else None
                baseline = st.session_state.personal_baselines.get(metric, {})
                st.metric("Current", f"{current_value:.1f}" if current_value is not None else "No Data")
                st.metric("Target", f"{baseline.get('target', 'N/A')}")
                if len(data.dropna()) > 1:
                    prev = data.dropna().iloc[-2]
                    change = (current_value - prev) / prev * 100 if prev != 0 else 0
                    st.metric("Change", f"{change:+.1f}%" )
                else:
                    st.metric("Change", "N/A")

    def create_alert_system(self):
        """Display warnings when metrics breach thresholds."""
        st.subheader("🚨 Real-Time Health Alerts")
        alerts = []
        for metric in st.session_state.metric_preferences['primary_metrics']:
            data = self.load_data_from_db(metric)
            if data.dropna().empty:
                alerts.append({"metric": metric, "msg": "No data available", "severity": "low"})
                continue
            current_value = data.dropna().iloc[-1]
            b = st.session_state.personal_baselines.get(metric, {})
            if current_value < b.get('min', -np.inf):
                alerts.append({"metric": metric, "msg": f"Below minimum ({b.get('min')})", "severity": "high"})
            elif current_value > b.get('max', np.inf):
                alerts.append({"metric": metric, "msg": f"Above maximum ({b.get('max')})", "severity": "high"})
            elif abs(current_value - b.get('target', 0)) <= max(1, 0.05 * b.get('target', 1)):
                alerts.append({"metric": metric, "msg": "At/near target", "severity": "low"})
        if alerts:
            for a in alerts:
                c = {"high": "🛑", "medium": "⚠️", "low": "✅"}[a['severity']]
                st.write(f"{c} **{a['metric']}**: {a['msg']}")
        else:
            st.info("✅ All metrics normal")

    def create_custom_metric_builder(self):

    # Display existing custom metrics
        if 'custom_metrics' in st.session_state and st.session_state.custom_metrics:
            st.write("**Your Custom Metrics:**")
            for name, details in st.session_state.custom_metrics.items():
                with st.expander(f"📊 {name}"):
                    st.code(details['formula'])
                    if st.button(f"Delete {name}", key=f"delete_{name}"):
                        del st.session_state.custom_metrics[name]
                        st.rerun()
    """Add your previous implementation for custom metrics if desired."""
        # Optional: let users build new metrics from database-backed metrics

