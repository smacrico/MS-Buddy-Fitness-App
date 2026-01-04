# PROPOSED ENHANCEMENTS FOR RunningAnalysis
# New calculations and visualizations using: avg_speed, max_speed, HR_RS_Deviation_Index
# (c) smacrico - Jan 2025

"""
NEW CALCULATIONS USING THE ADDED FIELDS:
==========================================

1. SPEED-BASED METRICS:
   - Speed Reserve: max_speed - avg_speed (sprint capacity)
   - Speed Consistency: avg_speed / max_speed ratio
   - Pace per km: 60 / avg_speed (min/km)
   - Speed Efficiency: avg_speed / heart_rate (speed per HR unit)
   - Running Economy at Speed: running_economy / avg_speed

2. HR-RS DEVIATION METRICS:
   - Cardiac Autonomic Balance: HR_RS_Deviation_Index trend analysis
   - Stress Index: HR_RS_Deviation_Index combined with TRIMP
   - Recovery Indicator: HR_RS_Deviation_Index change rate
   - Training Adaptation: HR_RS_Deviation_Index vs performance correlation

3. COMBINED METRICS:
   - Physiological Efficiency Score: (avg_speed / heart_rate) * (1 / HR_RS_Deviation_Index)
   - Speed-VO2max Index: avg_speed * vo2max
   - Performance Readiness: combines HR_RS_Deviation_Index, recovery score, speed reserve
   - Fatigue Index: HR_RS_Deviation_Index * cardiac_drift / avg_speed

4. TREND ANALYSIS:
   - Speed progression over time
   - HR-RS Deviation Index stability
   - Speed-HR relationship evolution
   - Max speed development

5. COMPARATIVE METRICS:
   - Speed zones distribution
   - HR efficiency at different speeds
   - Deviation index by training intensity
"""

# =====================================
# METHOD 1: Enhanced load_training_data
# =====================================

def load_training_data_enhanced(self):
    """Load training data with new speed and HR-RS fields"""
    try:
        conn = sqlite3.connect(r'c:/smakrykoDBs/Apex.db')
        query = """
        SELECT 
            date,
            COALESCE(running_economy, 0) as running_economy,
            COALESCE(vo2max, 0) as vo2max,
            COALESCE(distance, 0) as distance,
            COALESCE(time, 0) as time,
            COALESCE(heart_rate, 0) as heart_rate,
            COALESCE(avg_speed, 0) as avg_speed,
            COALESCE(max_speed, 0) as max_speed,
            COALESCE(HR_RS_Deviation_Index, 0) as hr_rs_deviation,
            COALESCE(cardiacdrift, 0) as cardiac_drift,
            COALESCE(running_economy / NULLIF(vo2max, 0), 0) AS efficiency_score,
            COALESCE(running_economy * (distance / NULLIF(time, 0)), 0) AS energy_cost,
            -- NEW CALCULATED FIELDS
            COALESCE(max_speed - avg_speed, 0) as speed_reserve,
            COALESCE(avg_speed / NULLIF(max_speed, 0), 0) as speed_consistency,
            COALESCE(60.0 / NULLIF(avg_speed, 0), 0) as pace_per_km,
            COALESCE(avg_speed / NULLIF(heart_rate, 0), 0) as speed_efficiency,
            COALESCE(running_economy / NULLIF(avg_speed, 0), 0) as economy_at_speed,
            COALESCE(avg_speed * vo2max, 0) as speed_vo2max_index
        FROM running_sessions
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])

        # Calculate additional derived metrics
        df['duration_min'] = df['time'] / 60
        
        # TRIMP calculation
        rest_hr = 60
        max_hr = 190
        df['hr_ratio'] = (df['heart_rate'] - rest_hr) / (max_hr - rest_hr)
        df['TRIMP'] = df['duration_min'] * df['hr_ratio']

        # Physiological Efficiency Score (avoid division by zero)
        df['physio_efficiency'] = np.where(
            (df['hr_rs_deviation'] > 0) & (df['heart_rate'] > 0),
            (df['avg_speed'] / df['heart_rate']) * (1 / df['hr_rs_deviation']),
            0
        )

        # Fatigue Index
        df['fatigue_index'] = np.where(
            df['avg_speed'] > 0,
            (df['hr_rs_deviation'] * df['cardiac_drift']) / df['avg_speed'],
            0
        )

        # Speed zones (example: slow < 10, moderate 10-14, fast > 14 km/h)
        df['speed_zone'] = pd.cut(
            df['avg_speed'],
            bins=[0, 10, 14, np.inf],
            labels=['Slow', 'Moderate', 'Fast']
        )

        # Calculate weekly metrics
        df['week'] = df['date'].dt.isocalendar().week
        weekly_trimp = df.groupby('week')['TRIMP'].sum().reset_index(name='weekly_trimp')
        
        # Acute and Chronic loads
        weekly_trimp['acute_load'] = weekly_trimp['weekly_trimp'].rolling(window=1).mean()
        weekly_trimp['chronic_load'] = weekly_trimp['weekly_trimp'].rolling(window=4).mean()
        weekly_trimp['acwr'] = weekly_trimp['acute_load'] / (weekly_trimp['chronic_load'] + 1e-8)
        
        self.training_log = df
        self.weekly_trimp = weekly_trimp

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


# =====================================
# METHOD 2: Speed Analysis
# =====================================

def analyze_speed_metrics(self):
    """Analyze speed-related metrics"""
    try:
        if self.training_log.empty:
            print("No data available")
            return None
        
        print("\n" + "="*80)
        print("SPEED METRICS ANALYSIS")
        print("="*80)
        
        # Overall speed statistics
        print("\nOverall Speed Statistics:")
        print(f"Average Speed (mean):     {self.training_log['avg_speed'].mean():.2f} km/h")
        print(f"Average Speed (std):      {self.training_log['avg_speed'].std():.2f} km/h")
        print(f"Max Speed (mean):         {self.training_log['max_speed'].mean():.2f} km/h")
        print(f"Max Speed (peak):         {self.training_log['max_speed'].max():.2f} km/h")
        print(f"Speed Reserve (mean):     {self.training_log['speed_reserve'].mean():.2f} km/h")
        print(f"Speed Consistency (mean): {self.training_log['speed_consistency'].mean():.2%}")
        print(f"Average Pace:             {self.training_log['pace_per_km'].mean():.2f} min/km")
        
        # Speed efficiency (speed per heart beat)
        print(f"\nSpeed Efficiency:         {self.training_log['speed_efficiency'].mean():.4f} km/h per bpm")
        print(f"Economy at Speed:         {self.training_log['economy_at_speed'].mean():.2f}")
        
        # Speed zone distribution
        print("\nSpeed Zone Distribution:")
        zone_counts = self.training_log['speed_zone'].value_counts()
        for zone, count in zone_counts.items():
            pct = (count / len(self.training_log)) * 100
            print(f"  {zone}: {count} sessions ({pct:.1f}%)")
        
        # Trend analysis (last 5 vs first 5 sessions)
        if len(self.training_log) >= 10:
            recent_avg = self.training_log.tail(5)['avg_speed'].mean()
            early_avg = self.training_log.head(5)['avg_speed'].mean()
            improvement = ((recent_avg - early_avg) / early_avg) * 100
            print(f"\nSpeed Improvement (recent vs early): {improvement:+.2f}%")
        
        return {
            'avg_speed_mean': self.training_log['avg_speed'].mean(),
            'max_speed_peak': self.training_log['max_speed'].max(),
            'speed_reserve': self.training_log['speed_reserve'].mean(),
            'speed_consistency': self.training_log['speed_consistency'].mean(),
            'pace_per_km': self.training_log['pace_per_km'].mean()
        }
    
    except Exception as e:
        print(f"Error in speed analysis: {e}")
        return None


# =====================================
# METHOD 3: HR-RS Deviation Analysis
# =====================================

def analyze_hr_rs_deviation(self):
    """Analyze HR-RS Deviation Index patterns"""
    try:
        if self.training_log.empty:
            print("No data available")
            return None
        
        print("\n" + "="*80)
        print("HR-RS DEVIATION INDEX ANALYSIS")
        print("="*80)
        
        # Filter out zero values
        valid_data = self.training_log[self.training_log['hr_rs_deviation'] > 0]
        
        if valid_data.empty:
            print("No HR-RS Deviation data available")
            return None
        
        print("\nOverall HR-RS Deviation Statistics:")
        print(f"Mean:                 {valid_data['hr_rs_deviation'].mean():.2f}")
        print(f"Std Dev:              {valid_data['hr_rs_deviation'].std():.2f}")
        print(f"Min:                  {valid_data['hr_rs_deviation'].min():.2f}")
        print(f"Max:                  {valid_data['hr_rs_deviation'].max():.2f}")
        
        # Calculate stability (coefficient of variation)
        cv = (valid_data['hr_rs_deviation'].std() / valid_data['hr_rs_deviation'].mean()) * 100
        print(f"Coefficient of Variation: {cv:.2f}% ", end="")
        if cv < 10:
            print("(Very Stable)")
        elif cv < 20:
            print("(Stable)")
        elif cv < 30:
            print("(Moderate Variability)")
        else:
            print("(High Variability)")
        
        # Trend analysis - calculate change rate
        if len(valid_data) >= 5:
            valid_data = valid_data.sort_values('date')
            recent_mean = valid_data.tail(3)['hr_rs_deviation'].mean()
            earlier_mean = valid_data.head(3)['hr_rs_deviation'].mean()
            change_rate = ((recent_mean - earlier_mean) / earlier_mean) * 100
            
            print(f"\nRecent Trend: {change_rate:+.2f}% ", end="")
            if abs(change_rate) < 5:
                print("(Stable)")
            elif change_rate > 5:
                print("(Increasing - may indicate fatigue)")
            else:
                print("(Decreasing - may indicate improved adaptation)")
        
        # Correlation with performance metrics
        if len(valid_data) >= 10:
            corr_speed = valid_data['hr_rs_deviation'].corr(valid_data['avg_speed'])
            corr_hr = valid_data['hr_rs_deviation'].corr(valid_data['heart_rate'])
            corr_vo2 = valid_data['hr_rs_deviation'].corr(valid_data['vo2max'])
            
            print("\nCorrelations with Performance:")
            print(f"  vs. Average Speed:  {corr_speed:+.3f}")
            print(f"  vs. Heart Rate:     {corr_hr:+.3f}")
            print(f"  vs. VO2max:         {corr_vo2:+.3f}")
        
        return {
            'mean': valid_data['hr_rs_deviation'].mean(),
            'std': valid_data['hr_rs_deviation'].std(),
            'stability_cv': cv
        }
    
    except Exception as e:
        print(f"Error in HR-RS deviation analysis: {e}")
        return None


# =====================================
# METHOD 4: Advanced Speed Visualizations
# =====================================

def visualize_speed_metrics(self):
    """Create comprehensive speed-related visualizations"""
    try:
        if self.training_log.empty:
            print("No data available")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Speed Metrics Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Average Speed Trend
        ax1 = axes[0, 0]
        ax1.plot(self.training_log['date'], self.training_log['avg_speed'], 
                marker='o', color='blue', label='Avg Speed')
        ax1.plot(self.training_log['date'], self.training_log['max_speed'], 
                marker='s', color='red', alpha=0.6, label='Max Speed')
        ax1.set_title('Speed Trends Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Speed (km/h)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Speed Reserve
        ax2 = axes[0, 1]
        ax2.plot(self.training_log['date'], self.training_log['speed_reserve'], 
                marker='o', color='green')
        ax2.set_title('Speed Reserve (Max - Avg)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Speed Reserve (km/h)')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Speed vs Heart Rate
        ax3 = axes[1, 0]
        scatter = ax3.scatter(self.training_log['heart_rate'], 
                            self.training_log['avg_speed'],
                            c=self.training_log['date'].astype(np.int64),
                            cmap='viridis', s=100, alpha=0.6)
        ax3.set_title('Speed vs Heart Rate (colored by time)')
        ax3.set_xlabel('Heart Rate (bpm)')
        ax3.set_ylabel('Average Speed (km/h)')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Date')
        
        # Plot 4: Speed Efficiency (Speed/HR)
        ax4 = axes[1, 1]
        ax4.plot(self.training_log['date'], self.training_log['speed_efficiency'], 
                marker='o', color='purple')
        ax4.set_title('Speed Efficiency (Speed per HR unit)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Speed/HR (km/h per bpm)')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # Plot 5: Pace Progression
        ax5 = axes[2, 0]
        ax5.plot(self.training_log['date'], self.training_log['pace_per_km'], 
                marker='o', color='orange')
        ax5.set_title('Pace Progression')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Pace (min/km)')
        ax5.invert_yaxis()  # Lower is better for pace
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Speed Zone Distribution
        ax6 = axes[2, 1]
        if 'speed_zone' in self.training_log.columns:
            zone_counts = self.training_log['speed_zone'].value_counts()
            ax6.bar(zone_counts.index.astype(str), zone_counts.values, 
                   color=['#3498db', '#2ecc71', '#e74c3c'])
            ax6.set_title('Training Sessions by Speed Zone')
            ax6.set_xlabel('Speed Zone')
            ax6.set_ylabel('Number of Sessions')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in speed visualization: {e}")


# =====================================
# METHOD 5: HR-RS Deviation Visualizations
# =====================================

def visualize_hr_rs_deviation(self):
    """Create HR-RS Deviation Index visualizations"""
    try:
        if self.training_log.empty:
            print("No data available")
            return
        
        # Filter valid data
        valid_data = self.training_log[self.training_log['hr_rs_deviation'] > 0].copy()
        
        if valid_data.empty:
            print("No HR-RS Deviation data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('HR-RS Deviation Index Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: HR-RS Deviation Trend
        ax1 = axes[0, 0]
        ax1.plot(valid_data['date'], valid_data['hr_rs_deviation'], 
                marker='o', color='red', linewidth=2)
        # Add rolling average
        rolling_avg = valid_data['hr_rs_deviation'].rolling(window=3, min_periods=1).mean()
        ax1.plot(valid_data['date'], rolling_avg, 
                linestyle='--', color='blue', linewidth=2, label='3-session avg')
        ax1.set_title('HR-RS Deviation Index Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Deviation Index')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Deviation vs Performance (Speed)
        ax2 = axes[0, 1]
        ax2.scatter(valid_data['hr_rs_deviation'], valid_data['avg_speed'],
                   s=100, alpha=0.6, c='green')
        ax2.set_title('HR-RS Deviation vs Speed Performance')
        ax2.set_xlabel('HR-RS Deviation Index')
        ax2.set_ylabel('Average Speed (km/h)')
        ax2.grid(True, alpha=0.3)
        
        # Add trend line
        if len(valid_data) >= 3:
            z = np.polyfit(valid_data['hr_rs_deviation'], valid_data['avg_speed'], 1)
            p = np.poly1d(z)
            ax2.plot(valid_data['hr_rs_deviation'], 
                    p(valid_data['hr_rs_deviation']), 
                    "r--", alpha=0.8, linewidth=2, label='Trend')
            ax2.legend()
        
        # Plot 3: Deviation Distribution
        ax3 = axes[1, 0]
        ax3.hist(valid_data['hr_rs_deviation'], bins=15, color='purple', 
                alpha=0.7, edgecolor='black')
        ax3.axvline(valid_data['hr_rs_deviation'].mean(), 
                   color='red', linestyle='--', linewidth=2, label='Mean')
        ax3.set_title('HR-RS Deviation Distribution')
        ax3.set_xlabel('Deviation Index')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Deviation vs TRIMP (if available)
        ax4 = axes[1, 1]
        if 'TRIMP' in valid_data.columns:
            ax4.scatter(valid_data['TRIMP'], valid_data['hr_rs_deviation'],
                       s=100, alpha=0.6, c='orange')
            ax4.set_title('HR-RS Deviation vs Training Load (TRIMP)')
            ax4.set_xlabel('TRIMP Score')
            ax4.set_ylabel('HR-RS Deviation Index')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'TRIMP data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error in HR-RS deviation visualization: {e}")


# =====================================
# METHOD 6: Combined Performance Dashboard
# =====================================

def create_performance_dashboard(self):
    """Create comprehensive dashboard with all new metrics"""
    try:
        if self.training_log.empty:
            print("No data available")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        fig.suptitle('Comprehensive Running Performance Dashboard', 
                    fontsize=18, fontweight='bold')
        
        # Row 1: Speed metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.training_log['date'], self.training_log['avg_speed'], 
                marker='o', color='blue')
        ax1.set_title('Average Speed Trend')
        ax1.set_ylabel('Speed (km/h)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.training_log['date'], self.training_log['speed_reserve'], 
                marker='o', color='green')
        ax2.set_title('Speed Reserve')
        ax2.set_ylabel('km/h')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.training_log['date'], self.training_log['pace_per_km'], 
                marker='o', color='orange')
        ax3.set_title('Pace')
        ax3.set_ylabel('min/km')
        ax3.invert_yaxis()
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Row 2: HR-RS Deviation
        valid_hr_rs = self.training_log[self.training_log['hr_rs_deviation'] > 0]
        
        ax4 = fig.add_subplot(gs[1, 0])
        if not valid_hr_rs.empty:
            ax4.plot(valid_hr_rs['date'], valid_hr_rs['hr_rs_deviation'], 
                    marker='o', color='red')
            ax4.set_title('HR-RS Deviation Index')
            ax4.set_ylabel('Index')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        if not valid_hr_rs.empty:
            ax5.scatter(valid_hr_rs['hr_rs_deviation'], valid_hr_rs['avg_speed'],
                       s=100, alpha=0.6, c='purple')
            ax5.set_title('Deviation vs Speed')
            ax5.set_xlabel('HR-RS Deviation')
            ax5.set_ylabel('Speed (km/h)')
            ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(gs[1, 2])
        if not valid_hr_rs.empty:
            ax6.hist(valid_hr_rs['hr_rs_deviation'], bins=15, 
                    color='purple', alpha=0.7, edgecolor='black')
            ax6.set_title('Deviation Distribution')
            ax6.set_xlabel('Index')
            ax6.grid(True, alpha=0.3, axis='y')
        
        # Row 3: Efficiency metrics
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(self.training_log['date'], self.training_log['speed_efficiency'], 
                marker='o', color='teal')
        ax7.set_title('Speed Efficiency (Speed/HR)')
        ax7.set_ylabel('km/h per bpm')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(self.training_log['date'], self.training_log['economy_at_speed'], 
                marker='o', color='brown')
        ax8.set_title('Economy at Speed')
        ax8.set_ylabel('RE / Speed')
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(True, alpha=0.3)
        
        ax9 = fig.add_subplot(gs[2, 2])
        if 'physio_efficiency' in self.training_log.columns:
            valid_physio = self.training_log[self.training_log['physio_efficiency'] > 0]
            if not valid_physio.empty:
                ax9.plot(valid_physio['date'], valid_physio['physio_efficiency'], 
                        marker='o', color='darkgreen')
                ax9.set_title('Physiological Efficiency')
                ax9.set_ylabel('Composite Score')
                ax9.tick_params(axis='x', rotation=45)
                ax9.grid(True, alpha=0.3)
        
        # Row 4: Combined analysis
        ax10 = fig.add_subplot(gs[3, :2])
        ax10_twin = ax10.twinx()
        
        line1 = ax10.plot(self.training_log['date'], self.training_log['avg_speed'], 
                         marker='o', color='blue', label='Avg Speed')
        line2 = ax10_twin.plot(self.training_log['date'], self.training_log['heart_rate'], 
                              marker='s', color='red', alpha=0.6, label='Heart Rate')
        
        ax10.set_title('Speed vs Heart Rate Over Time')
        ax10.set_xlabel('Date')
        ax10.set_ylabel('Speed (km/h)', color='blue')
        ax10_twin.set_ylabel('Heart Rate (bpm)', color='red')
        ax10.tick_params(axis='x', rotation=45)
        ax10.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax10.legend(lines, labels, loc='upper left')
        
        ax11 = fig.add_subplot(gs[3, 2])
        if 'speed_zone' in self.training_log.columns:
            zone_counts = self.training_log['speed_zone'].value_counts()
            colors = ['#3498db', '#2ecc71', '#e74c3c']
            ax11.pie(zone_counts.values, labels=zone_counts.index, 
                    autopct='%1.1f%%', colors=colors, startangle=90)
            ax11.set_title('Speed Zone Distribution')
        
        plt.savefig('c:/temp/logsFitnessApp/performance_dashboard.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nDashboard saved to: c:/temp/logsFitnessApp/performance_dashboard.png")
        
    except Exception as e:
        print(f"Error creating dashboard: {e}")


# =====================================
# SUMMARY OF PROPOSED ADDITIONS
# =====================================

"""
TO INTEGRATE INTO RunningAnalysis_v6.25.py:

1. Replace load_training_data() with load_training_data_enhanced()
   - Adds: avg_speed, max_speed, hr_rs_deviation
   - Calculates: speed_reserve, speed_consistency, pace_per_km, 
                 speed_efficiency, economy_at_speed, physio_efficiency,
                 fatigue_index, speed_zones

2. Add new analysis methods:
   - analyze_speed_metrics()
   - analyze_hr_rs_deviation()

3. Add new visualization methods:
   - visualize_speed_metrics()
   - visualize_hr_rs_deviation()
   - create_performance_dashboard()

4. Update monthly_summaries table to include:
   - avg_speed_mean, avg_speed_std
   - max_speed_mean, max_speed_std
   - speed_reserve_mean, speed_reserve_std
   - hr_rs_deviation_mean, hr_rs_deviation_std
   - speed_efficiency_mean, speed_efficiency_std

5. Update metrics_breakdown table to include new metrics

6. Add to main() function:
   - Call analyze_speed_metrics()
   - Call analyze_hr_rs_deviation()
   - Call visualize_speed_metrics()
   - Call visualize_hr_rs_deviation()
   - Call create_performance_dashboard()
"""
