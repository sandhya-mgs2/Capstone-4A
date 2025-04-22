import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import xgboost as xgb
import os

# --- Constants and Configuration ---
APP1_REGION_MAPPING = {
    'Alabama': 'South', 'Alaska': 'West', 'Arizona': 'West', 'Arkansas': 'South',
    'California': 'West', 'Colorado': 'West', 'Connecticut': 'Northeast',
    'Delaware': 'South', 'District of Columbia': 'South', 'Florida': 'South',
    'Georgia': 'South', 'Hawaii': 'West', 'Idaho': 'West', 'Illinois': 'Midwest',
    'Indiana': 'Midwest', 'Iowa': 'Midwest', 'Kansas': 'Midwest', 'Kentucky': 'South',
    'Louisiana': 'South', 'Maine': 'Northeast', 'Maryland': 'South',
    'Massachusetts': 'Northeast', 'Michigan': 'Midwest', 'Minnesota': 'Midwest',
    'Mississippi': 'South', 'Missouri': 'Midwest', 'Montana': 'West',
    'Nebraska': 'Midwest', 'Nevada': 'West', 'New Hampshire': 'Northeast',
    'New Jersey': 'Northeast', 'New Mexico': 'West', 'New York': 'Northeast',
    'North Carolina': 'South', 'North Dakota': 'Midwest', 'Ohio': 'Midwest',
    'Oklahoma': 'South', 'Oregon': 'West', 'Pennsylvania': 'Northeast',
    'Rhode Island': 'Northeast', 'South Carolina': 'South', 'South Dakota': 'Midwest',
    'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West', 'Vermont': 'Northeast',
    'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South',
    'Wisconsin': 'Midwest', 'Wyoming': 'West'
}
APP1_STATE_ABBREV = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI',
    'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
    'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}
APP1_CATEGORIES = {
    'symptom_searches': {
        'display_name': 'Symptom Searches',
        'indicators': ['depression symptoms', 'signs of depression', 'feeling hopeless', 'feeling worthless',
                       'feeling empty']
    },
    'diagnostic_searches': {
        'display_name': 'Diagnostic Searches',
        'indicators': ['am i depressed', 'depression test', 'depression quiz', 'depression screening',
                       'depression assessment']
    },
    'physical_searches': {
        'display_name': 'Physical Symptoms Searches',
        'indicators': ['depression fatigue', 'always tired', 'constant fatigue', 'depression physical symptoms',
                       'no energy to do anything']
    },
    'coping_searches': {
        'display_name': 'Coping & Management Searches',
        'indicators': ['depression self help', 'depression coping', 'depression management', 'depression exercises',
                       'depression self care']
    }
}
APP1_COLORS = {
    'primary': '#1A2B4C', 'secondary': '#4A5B7C', 'accent': '#3498db',
    'background': '#f8f9fa', 'card': '#ffffff', 'text': '#333333',
    'muted': '#6c757d', 'success': '#28a745', 'warning': '#ffc107',
    'danger': '#dc3545', 'info': '#17a2b8', 'border': '#dee2e6'
}
APP1_COLOR_SCALE = 'Viridis'
APP1_COLOR_SCALE_REVERSE = 'RdBu_r'
APP1_MAP_COLORS = {'lake': 'rgb(255,255,255)', 'land': 'rgb(240,240,240)', 'subunit': 'rgb(180,180,180)'}
APP1_CARD_STYLE = {'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'borderRadius': '8px', 'border': 'none', 'marginBottom': '24px', 'transition': 'transform 0.3s ease'}
APP1_CARD_HEADER_STYLE = {'backgroundColor': APP1_COLORS['primary'], 'color': 'white', 'fontWeight': 'bold', 'padding': '12px 20px', 'borderTopLeftRadius': '8px', 'borderTopRightRadius': '8px'}
APP1_GRAPH_STYLE = {'borderRadius': '0 0 8px 8px', 'padding': '8px'}

APP3_MODEL_PATH = 'best_xgboost_mental_health_model.pkl'
APP3_REGIONS = ['Northeast', 'Midwest', 'South', 'West', 'Territory']
APP3_MONTHS = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
APP3_COLORS = {
    'primary': '#2c3e50', 'secondary': '#3498db', 'anxiety': '#3498db',
    'depression': '#9b59b6', 'background': '#f8f9fa', 'card': '#ffffff',
    'text': '#333333', 'border': '#dddddd', 'success': '#2ecc71',
    'warning': '#f39c12', 'error': '#e74c3c'
}
APP3_CARD_STYLE = {'backgroundColor': APP3_COLORS['card'], 'borderRadius': '8px', 'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)', 'padding': '24px', 'marginBottom': '24px'}
APP3_HEADER_STYLE = {'color': APP3_COLORS['primary'], 'marginBottom': '16px', 'paddingBottom': '8px', 'borderBottom': f'2px solid {APP3_COLORS["secondary"]}', 'fontWeight': 'bold'}
APP3_LABEL_STYLE = {'color': APP3_COLORS['primary'], 'fontWeight': 'bold', 'marginTop': '16px', 'marginBottom': '8px', 'display': 'block'}
APP3_BUTTON_STYLE = {'backgroundColor': APP3_COLORS['secondary'], 'color': 'white', 'border': 'none', 'padding': '12px 24px', 'borderRadius': '4px', 'cursor': 'pointer', 'fontSize': '16px', 'marginTop': '24px', 'width': '100%', 'transition': 'background-color 0.3s', 'fontWeight': 'bold'}

# --- Helper Functions & Classes ---

# App 1: Data Processor
class App1DataProcessor:
    def __init__(self, file_path='mental_health_trends_all_states.csv'):
        self.df = self._load_and_process_data(file_path)
        if self.df is not None:
            self.top_states = self.df.nlargest(5, 'depression_index')[['geoName', 'depression_index']]
            self.bottom_states = self.df.nsmallest(5, 'depression_index')[['geoName', 'depression_index']]
            self.region_stats = self._compute_region_stats()
        else:
            self.top_states = pd.DataFrame(columns=['geoName', 'depression_index'])
            self.bottom_states = pd.DataFrame(columns=['geoName', 'depression_index'])
            self.region_stats = pd.DataFrame(columns=['region', 'depression_index'])

    def _load_and_process_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df['region'] = df['geoName'].map(APP1_REGION_MAPPING)
            df['state_code'] = df['geoName'].map(APP1_STATE_ABBREV)
            for category, config in APP1_CATEGORIES.items():
                df[category] = df[config['indicators']].mean(axis=1)
            exclude_cols = ['geoName', 'region', 'state_code'] + list(APP1_CATEGORIES.keys())
            indicator_cols = [col for col in df.columns if col not in exclude_cols]
            df['depression_index'] = df[indicator_cols].mean(axis=1)
            return df
        except FileNotFoundError:
            print(f"Warning: App1 data file '{file_path}' not found. App1 features will be limited.")
            # Optionally create sample data here if needed for basic functionality
            return None
        except Exception as e:
            print(f"Error loading App1 data: {e}")
            return None

    def _compute_region_stats(self):
        if self.df is None:
            return pd.DataFrame(columns=['region', 'depression_index'])
        return (self.df.groupby('region')['depression_index']
                .mean()
                .reset_index()
                .sort_values('depression_index', ascending=False))

    def get_dropdown_options(self):
        if self.df is None:
            return [{'label': 'Overall Depression Index', 'value': 'depression_index'}]
        all_indicators = [c for c in self.df.columns
                          if c not in ['geoName', 'region', 'state_code', 'depression_index'] + list(APP1_CATEGORIES.keys())]
        dropdown_options = [{'label': c, 'value': c} for c in all_indicators]
        dropdown_options.insert(0, {'label': 'Overall Depression Index', 'value': 'depression_index'})
        for i, (category, config) in enumerate(APP1_CATEGORIES.items(), 1):
            dropdown_options.insert(i, {'label': config['display_name'], 'value': category})
        return dropdown_options

    def get_state_options(self):
        if self.df is None:
            return []
        return [{'label': s, 'value': s} for s in self.df['geoName']]

# App 2: Data Loading and Plotting
def app2_load_data(file_path='combined_depression_anxiety.csv'):
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))
        df['Month_Name'] = df['Date'].dt.strftime('%B')
        df['Year_Month'] = df['Date'].dt.strftime('%Y-%m')
        state_to_abbrev = APP1_STATE_ABBREV # Reuse from App1
        df['State_Code'] = df['State'].map(state_to_abbrev)
        df = df[(df['Combined_Value'] >= 5) & (df['Combined_Value'] <= 60)]
        mainland_states = [state for state in df['State'].unique()
                           if state not in ['Guam', 'Puerto Rico', 'Virgin Islands']]
        df_mainland = df[df['State'].isin(mainland_states)].copy() # Use copy to avoid SettingWithCopyWarning
        return df_mainland
    except FileNotFoundError:
        print(f"Warning: App2 data file '{file_path}' not found. App2 features will be limited.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        print(f"Error loading App2 data: {e}")
        return pd.DataFrame()

def app2_create_animated_choropleth(df, indicator=None, colorscale='RdBu_r', frame_duration=800):
    if df.empty: return go.Figure() # Return empty figure if no data
    if indicator and indicator != 'combined':
        filtered_df = df[df['Indicated'] == indicator]
    else:
        filtered_df = df
    filtered_df = filtered_df.sort_values('Date')
    fig = px.choropleth(
        filtered_df, locations='State_Code', locationmode='USA-states',
        color='Combined_Value', animation_frame='Year_Month',
        color_continuous_scale=colorscale, scope='usa', range_color=[10, 50],
        labels={'Combined_Value': 'Search Interest'}, hover_data=['State'],
        title=f"{'Depression & Anxiety' if not indicator or indicator == 'combined' else indicator} Google Trends Over Time (2020-2024)"
    )
    fig.update_layout(height=700, geo=dict(showlakes=True, lakecolor='rgb(255, 255, 255)', subunitcolor='rgb(217, 217, 217)'),
                      coloraxis_colorbar=dict(title='Search<br>Interest', thicknessmode="pixels", thickness=20, lenmode="pixels", len=400))
    if fig.layout.updatemenus:
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = frame_duration
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 300
    if fig.layout.sliders:
        fig.layout.sliders[0].pad.t = 10
    return fig

def app2_create_top_states_chart(df, indicator=None, top_n=5):
    if df.empty: return go.Figure() # Return empty figure if no data
    if indicator and indicator != 'combined':
        filtered_df = df[df['Indicated'] == indicator]
    else:
        filtered_df = df
    top_states = filtered_df.groupby('State')['Combined_Value'].mean().nlargest(top_n).index
    top_states_df = filtered_df[filtered_df['State'].isin(top_states)]
    monthly_avg = top_states_df.groupby(['State', 'Year_Month', 'Date'])['Combined_Value'].mean().reset_index()
    monthly_avg = monthly_avg.sort_values(['State', 'Date'])
    fig = px.line(
        monthly_avg, x='Date', y='Combined_Value', color='State',
        labels={'Combined_Value': 'Search Interest', 'Date': 'Date', 'State': 'State'},
        title=f"Top {top_n} States - {'Depression & Anxiety' if not indicator or indicator == 'combined' else indicator} Search Interest"
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Search Interest (0-100)", legend_title="State", hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(buttons=list([dict(count=6, label="6m", step="month", stepmode="backward"),
                                                      dict(count=1, label="1y", step="year", stepmode="backward"),
                                                      dict(step="all")])))
    return fig

# App 3: Model Loading and Plotting
def app3_load_model(model_path=APP3_MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Note: Model file '{model_path}' not found. Using a placeholder model.")
        model = xgb.XGBRegressor()
        # Simple fit for placeholder
        try:
            model.fit(np.array([[2023, 6, 2023**2, np.sin(2*np.pi*6/12), np.cos(2*np.pi*6/12), 2023*np.sin(2*np.pi*6/12), 2023*np.cos(2*np.pi*6/12), 1, 0, 0, 0, 0, 1]]), np.array([50.0]))
            # Add feature names if possible after fit
            model.get_booster().feature_names = ['Year', 'Month', 'Year_Squared', 'Sin_Month', 'Cos_Month', 'Year_Sin', 'Year_Cos', 'Region_Northeast', 'Region_Midwest', 'Region_South', 'Region_West', 'Region_Territory', 'Indicated_Anxiety']
        except Exception as e:
            print(f"Could not fit placeholder model: {e}")
        # Save the placeholder if needed for consistency, though not strictly necessary if loaded in memory
        # joblib.dump(model, model_path)
        return model
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        # Fallback to placeholder if loading fails
        model = xgb.XGBRegressor()
        try:
            model.fit(np.array([[2023, 6, 2023**2, np.sin(2*np.pi*6/12), np.cos(2*np.pi*6/12), 2023*np.sin(2*np.pi*6/12), 2023*np.cos(2*np.pi*6/12), 1, 0, 0, 0, 0, 1]]), np.array([50.0]))
            model.get_booster().feature_names = ['Year', 'Month', 'Year_Squared', 'Sin_Month', 'Cos_Month', 'Year_Sin', 'Year_Cos', 'Region_Northeast', 'Region_Midwest', 'Region_South', 'Region_West', 'Region_Territory', 'Indicated_Anxiety']
        except Exception as fit_e:
             print(f"Could not fit placeholder model after load error: {fit_e}")
        return model


def app3_create_example_data(year, month, is_anxiety, region, model, regions_list):
    example = pd.DataFrame({
        'Year': [year], 'Month': [month], 'Year_Squared': [year ** 2],
        'Sin_Month': [np.sin(2 * np.pi * month / 12)], 'Cos_Month': [np.cos(2 * np.pi * month / 12)],
        'Year_Sin': [year * np.sin(2 * np.pi * month / 12)], 'Year_Cos': [year * np.cos(2 * np.pi * month / 12)]
    })
    try:
        # Attempt to get feature names from the loaded model
        feature_names = model.get_booster().feature_names
    except Exception:
        # Fallback if feature names aren't available (e.g., placeholder or older model)
        print("Warning: Could not get feature names from model. Using default feature set.")
        feature_names = ['Year', 'Month', 'Year_Squared', 'Sin_Month', 'Cos_Month', 'Year_Sin', 'Year_Cos']
        for r in regions_list: feature_names.append(f'Region_{r}')
        feature_names.append('Indicated_Anxiety')

    # Add indicator columns based on expected feature names
    for col in feature_names:
        if col.startswith('Region_'):
            region_name = col.replace('Region_', '')
            example[col] = [1 if region_name == region else 0]
        elif col == 'Indicated_Anxiety':
            example[col] = [1 if is_anxiety else 0]
        elif col not in example.columns:
            # Add missing columns expected by the model, initialized to 0
            example[col] = 0

    # Ensure columns are in the order expected by the model
    try:
        return example[feature_names]
    except KeyError as e:
        print(f"KeyError preparing features: {e}. Columns available: {list(example.columns)}. Expected: {feature_names}")
        # Return with available columns, prediction might fail
        return example


def app3_create_gauge_chart(value, condition):
    max_value = 100
    if value > max_value: max_value = value * 1.2
    color = APP3_COLORS['anxiety'] if condition == 'Anxiety' else APP3_COLORS['depression']
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{condition} Score", 'font': {'size': 24, 'color': APP3_COLORS['primary']}},
        gauge={'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': APP3_COLORS['primary']},
               'bar': {'color': color}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': APP3_COLORS['border'],
               'steps': [{'range': [0, max_value / 3], 'color': '#e8f8f5'},
                         {'range': [max_value / 3, 2 * max_value / 3], 'color': '#d1f5ee'},
                         {'range': [2 * max_value / 3, max_value], 'color': '#a9e2d9'}]}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor=APP3_COLORS['card'], font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'})
    return fig

def app3_create_monthly_forecast(year, region, condition_text, is_anxiety, model, regions_list, months_dict):
    months_list = list(range(1, 13))
    predictions = []
    for month in months_list:
        features = app3_create_example_data(year, month, is_anxiety, region, model, regions_list)
        try:
            prediction = model.predict(features)[0]
            predictions.append(prediction)
        except Exception as e:
            print(f"Error predicting for month {month}: {e}")
            predictions.append(np.nan) # Append NaN if prediction fails

    fig = go.Figure()
    color = APP3_COLORS['anxiety'] if condition_text == 'Anxiety' else APP3_COLORS['depression']
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    avg_prediction = sum(valid_predictions) / len(valid_predictions) if valid_predictions else 0

    fig.add_trace(go.Scatter(x=[months_dict[m] for m in months_list], y=predictions, mode='lines+markers', name=condition_text, line=dict(color=color, width=3), marker=dict(size=10, color=color)))
    fig.add_trace(go.Scatter(x=[months_dict[m] for m in months_list], y=[avg_prediction] * len(months_list), mode='lines', name='Average', line=dict(color=APP3_COLORS['error'], width=2, dash='dash')))
    fig.update_layout(title={'text': f"Monthly {condition_text} Score Forecast for {region} in {year}", 'font': {'color': APP3_COLORS['primary']}},
                      xaxis_title="Month", yaxis_title="Predicted Score", legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                      paper_bgcolor=APP3_COLORS['card'], plot_bgcolor=APP3_COLORS['card'], font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'},
                      height=400, margin=dict(l=20, r=20, t=50, b=40))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eeeeee')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eeeeee')
    return fig

def app3_create_regional_comparison(year, month, is_anxiety, condition_text, model, regions_list, months_dict):
    predictions = {}
    for region in regions_list:
        features = app3_create_example_data(year, month, is_anxiety, region, model, regions_list)
        try:
            prediction = model.predict(features)[0]
            predictions[region] = prediction
        except Exception as e:
            print(f"Error predicting for region {region}: {e}")
            predictions[region] = np.nan # Assign NaN if prediction fails

    valid_predictions = {r: p for r, p in predictions.items() if not np.isnan(p)}
    avg = sum(valid_predictions.values()) / len(valid_predictions) if valid_predictions else 0

    fig = go.Figure()
    color = APP3_COLORS['anxiety'] if condition_text == 'Anxiety' else APP3_COLORS['depression']
    fig.add_trace(go.Bar(x=list(predictions.keys()), y=list(predictions.values()), marker_color=color, text=[f"{value:.2f}" if not np.isnan(value) else "N/A" for value in predictions.values()], textposition='auto'))
    fig.add_shape(type="line", x0=-0.5, y0=avg, x1=len(regions_list) - 0.5, y1=avg, line=dict(color=APP3_COLORS['error'], width=2, dash="dash"))
    fig.add_annotation(x=len(regions_list) - 0.7, y=avg, text=f"Avg: {avg:.2f}", showarrow=False, font=dict(color=APP3_COLORS['error']))
    fig.update_layout(title={'text': f"Regional Comparison of {condition_text} Scores ({months_dict[month]} {year})", 'font': {'color': APP3_COLORS['primary']}},
                      xaxis_title="Region", yaxis_title="Predicted Score", paper_bgcolor=APP3_COLORS['card'], plot_bgcolor=APP3_COLORS['card'],
                      font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'}, height=400, margin=dict(l=20, r=20, t=50, b=40))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eeeeee')
    return fig

def app3_create_empty_gauge():
    fig = go.Figure(go.Indicator(mode="gauge+number", value=0, domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Score", 'font': {'size': 24, 'color': APP3_COLORS['primary']}}, gauge={'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': APP3_COLORS['primary']}, 'bar': {'color': "lightgray"}, 'bgcolor': "white", 'borderwidth': 2, 'bordercolor': APP3_COLORS['border']}))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor=APP3_COLORS['card'], font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'})
    return fig

def app3_create_empty_forecast():
    fig = go.Figure()
    fig.update_layout(title={'text': "Monthly Forecast (Make a prediction first)", 'font': {'color': APP3_COLORS['primary']}}, xaxis_title="Month", yaxis_title="Predicted Score", paper_bgcolor=APP3_COLORS['card'], plot_bgcolor=APP3_COLORS['card'], font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'}, height=400, margin=dict(l=20, r=20, t=50, b=40))
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Generate a prediction to see monthly forecast", showarrow=False, font=dict(color=APP3_COLORS['secondary'], size=16))
    return fig

def app3_create_empty_comparison():
    fig = go.Figure()
    fig.update_layout(title={'text': "Regional Comparison (Make a prediction first)", 'font': {'color': APP3_COLORS['primary']}}, xaxis_title="Region", yaxis_title="Predicted Score", paper_bgcolor=APP3_COLORS['card'], plot_bgcolor=APP3_COLORS['card'], font={'color': APP3_COLORS['text'], 'family': 'Roboto, sans-serif'}, height=400, margin=dict(l=20, r=20, t=50, b=40))
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text="Generate a prediction to see regional comparison", showarrow=False, font=dict(color=APP3_COLORS['secondary'], size=16))
    return fig

# --- Layout Creation Functions ---

def create_app1_layout(data_processor):
    if data_processor.df is None:
        return dbc.Container([dbc.Alert("App1 data could not be loaded. Please check the file 'mental_health_trends_all_states.csv'.", color="danger")])

    # Styles specific to App1 layout items
    app1_insights_card_style = {'maxHeight': '345px', 'overflowY': 'auto'}
    app1_insight_item_style = {'backgroundColor': APP1_COLORS['background'], 'padding': '10px 15px', 'borderRadius': '4px', 'marginBottom': '10px', 'borderLeft': f'4px solid {APP1_COLORS["accent"]}'}
    app1_state_name_style = {'fontWeight': 'bold', 'color': APP1_COLORS['primary']}
    app1_index_value_style = {'backgroundColor': '#e9ecef', 'padding': '2px 8px', 'borderRadius': '4px', 'fontWeight': '500'}

    return dbc.Container([
        dbc.Alert([html.I(className="bi bi-info-circle-fill me-2"), "Analysis of depression-related search trends (2024). Use the dropdown to explore indicators."], color="info", dismissable=True, className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.H4("Interactive Map", className="m-0 d-flex align-items-center"), html.I(className="bi bi-map ms-2")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(md=6, children=[
                            html.Label("Select indicator:", className="fw-bold mb-2"),
                            dcc.Dropdown(id='app1-indicator-dropdown', options=data_processor.get_dropdown_options(), value='depression_index', clearable=False, className="mb-3")
                        ]),
                        dbc.Col(md=6, children=[
                            html.Div([
                                html.H5("Search Categories:", className="fw-bold mb-2"),
                                html.Ul([html.Li([html.Span(APP1_CATEGORIES[cat]['display_name'], style={'fontWeight': 'bold'}), f": {', '.join(APP1_CATEGORIES[cat]['indicators'][:2])}..."]) for cat in APP1_CATEGORIES], className="small list-unstyled")
                            ], style={'backgroundColor': APP1_COLORS['background'], 'padding': '15px', 'borderRadius': '4px', 'height': '100%'})
                        ])
                    ], className="mb-3"),
                    dcc.Graph(id='app1-choropleth-map', style=APP1_GRAPH_STYLE, config={'displayModeBar': True, 'displaylogo': False, 'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian']})
                ])
            ], style=APP1_CARD_STYLE))
        ]),
        html.Hr(style={'margin': '30px 0'}),
        dbc.Row([
            dbc.Col(md=8, children=dbc.Card([
                dbc.CardHeader([html.H4("Categories Comparison", className="m-0"), html.I(className="bi bi-grid-3x3-gap")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                dbc.CardBody(dcc.Graph(id='app1-categories-graph', style=APP1_GRAPH_STYLE, config={'displayModeBar': True, 'displaylogo': False}))
            ], style=APP1_CARD_STYLE)),
            dbc.Col(md=4, children=[
                dbc.Card([
                    dbc.CardHeader([html.H4("Regional Analysis", className="m-0"), html.I(className="bi bi-bar-chart")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                    dbc.CardBody(dcc.Graph(id='app1-region-bar', style=APP1_GRAPH_STYLE, config={'displayModeBar': False}))
                ], style=APP1_CARD_STYLE),
                dbc.Card([
                    dbc.CardHeader([html.H4("Key Insights", className="m-0"), html.I(className="bi bi-lightbulb")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                    dbc.CardBody(html.Div([
                        html.H5("Highest Trends:", className="fw-bold mb-3"),
                        html.Div([html.Div([html.Span(f"{i+1}. ", className="me-2"), html.Span(f"{r['geoName']}", style=app1_state_name_style, className="me-2"), html.Span(f"{r['depression_index']:.2f}", style=app1_index_value_style)], style=app1_insight_item_style) for i, (_, r) in enumerate(data_processor.top_states.iterrows())]),
                        html.H5("Lowest Trends:", className="fw-bold mb-3 mt-4"),
                        html.Div([html.Div([html.Span(f"{i+1}. ", className="me-2"), html.Span(f"{r['geoName']}", style=app1_state_name_style, className="me-2"), html.Span(f"{r['depression_index']:.2f}", style=app1_index_value_style)], style=app1_insight_item_style) for i, (_, r) in enumerate(data_processor.bottom_states.iterrows())])
                    ], style=app1_insights_card_style))
                ], style=APP1_CARD_STYLE)
            ])
        ]),
        html.Hr(style={'margin': '30px 0'}),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.H4("State Comparison", className="m-0"), html.I(className="bi bi-arrow-left-right")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(md=6, children=[html.Label("Select first state:", className="fw-bold mb-2"), dcc.Dropdown(id='app1-state1-dropdown', options=data_processor.get_state_options(), value='California', clearable=False, className="mb-3")]),
                        dbc.Col(md=6, children=[html.Label("Select second state:", className="fw-bold mb-2"), dcc.Dropdown(id='app1-state2-dropdown', options=data_processor.get_state_options(), value='New York', clearable=False, className="mb-3")])
                    ]),
                    dcc.Graph(id='app1-state-comparison-graph', style=APP1_GRAPH_STYLE, config={'displayModeBar': True, 'displaylogo': False})
                ])
            ], style=APP1_CARD_STYLE))
        ]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader([html.H4("Methodology", className="m-0"), html.I(className="bi bi-info-circle")], style=APP1_CARD_HEADER_STYLE, className="d-flex justify-content-between"),
                dbc.CardBody([
                    html.P(["Data from ", html.Strong("Google Trends API"), " (2024). Normalized search volumes for depression terms."]),
                    html.P("Terms grouped into categories. Overall index is average of all terms."),
                    html.P("Regional classifications: US Census Bureau.")
                ])
            ], style=APP1_CARD_STYLE))
        ])
    ], fluid=True, className="pt-4 pb-5")

def create_app2_layout(df_mainland):
    if df_mainland.empty:
        return dbc.Container([dbc.Alert("App2 data could not be loaded. Please check the file 'combined_depression_anxiety.csv'.", color="danger")])

    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Depression & Anxiety Trends Across the US (2020-2024)", className="text-center my-4")
            ])
        ]),
        dbc.Row([
            # Controls Column
            dbc.Col(md=4, lg=3, children=[
                dbc.Card([
                    dbc.CardHeader("Visualization Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Data Type:", html_for="app2-indicator-selector"),
                        dbc.RadioItems(
                            id='app2-indicator-selector',
                            options=[{'label': 'Depression', 'value': 'Depression'},
                                     {'label': 'Anxiety', 'value': 'Anxiety'},
                                     {'label': 'Combined', 'value': 'combined'}],
                            value='combined', inline=False, className="mb-3",
                        ),
                        dbc.Label("Color Scale:", html_for="app2-colorscale-selector"),
                        dcc.Dropdown(
                            id='app2-colorscale-selector',
                            options=[{'label': 'Red-Blue', 'value': 'RdBu_r'}, {'label': 'YlOrRd', 'value': 'YlOrRd'},
                                     {'label': 'Blues', 'value': 'Blues'}, {'label': 'Reds', 'value': 'Reds'},
                                     {'label': 'Viridis', 'value': 'Viridis'}, {'label': 'Plasma', 'value': 'Plasma'}],
                            value='RdBu_r', clearable=False, className="mb-3"
                        ),
                        dbc.Label("Animation Speed (ms/frame):", html_for="app2-speed-slider"),
                        dcc.Slider(
                            id='app2-speed-slider', min=200, max=2000, step=100,
                            marks={200: 'Fast', 1000: 'Med', 2000: 'Slow'}, value=800, className="mb-4"
                        ),
                        dbc.Button("Update Map", id="app2-generate-button", color="primary", className="w-100")
                    ])
                ], className="mb-4") # Added margin bottom to card
            ]),
            # Map Column
            dbc.Col(md=8, lg=9, children=[
                dbc.Card([
                    dbc.CardHeader(id="app2-map-title", children="Animated Map: Depression & Anxiety Search Interest"),
                    dbc.CardBody(dcc.Loading(id="app2-loading-map", type="circle", children=[
                        dcc.Graph(id='app2-animated-map', style={'height': '650px'}) # Slightly adjusted height
                    ]))
                ])
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Top States Analysis"),
                    dbc.CardBody(dcc.Graph(id='app2-top-states-chart', style={'height': '450px'})) # Slightly adjusted height
                ], className="mt-4")
            ])
        ]),
        dbc.Row([
            dbc.Col(html.Div([
                html.Hr(),
                html.P("Dashboard visualizes Google Trends for depression/anxiety searches (2020-2024).", className="text-muted"),
                html.P("Use map controls to see trends over time.", className="text-muted")
            ], className="mt-4 text-center"))
        ])
    ], fluid=True, className="pt-4 pb-5")

def create_app3_layout():
    # Styles specific to App3 layout items
    app3_prediction_output_style = {'fontSize': '18px', 'marginBottom': '24px', 'padding': '16px', 'backgroundColor': '#e8f4f8', 'borderRadius': '4px', 'minHeight': '50px', 'border': f'1px solid {APP3_COLORS["secondary"]}'}
    app3_info_card_style = {**APP3_CARD_STYLE, 'backgroundColor': '#f0f4f8'} # Slightly different background for info

    return html.Div([
        html.Div([
            html.H1("Mental Health Score Predictor", style={'textAlign': 'center', 'color': APP3_COLORS['primary'], 'padding': '24px 0'})
        ], style={'borderBottom': f'3px solid {APP3_COLORS["secondary"]}', 'marginBottom': '32px'}),
        html.Div([
            # Left sidebar
            html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H2("Input Parameters", style=APP3_HEADER_STYLE)),
                    dbc.CardBody([
                        html.Label("Year:", style=APP3_LABEL_STYLE),
                        dcc.Slider(id='app3-year-slider', min=2020, max=2030, step=1, value=2025, marks={i: str(i) for i in range(2020, 2031, 2)}, tooltip={"placement": "bottom", "always_visible": True}, className='slider mb-3'),
                        html.Label("Month:", style=APP3_LABEL_STYLE),
                        dcc.Dropdown(id='app3-month-dropdown', options=[{'label': m, 'value': n} for n, m in APP3_MONTHS.items()], value=4, clearable=False, style={'borderRadius': '4px', 'border': f'1px solid {APP3_COLORS["border"]}', 'marginBottom': '16px'}),
                        html.Label("Region:", style=APP3_LABEL_STYLE),
                        dcc.Dropdown(id='app3-region-dropdown', options=[{'label': r, 'value': r} for r in APP3_REGIONS], value='Northeast', clearable=False, style={'borderRadius': '4px', 'border': f'1px solid {APP3_COLORS["border"]}', 'marginBottom': '16px'}),
                        html.Label("Condition:", style=APP3_LABEL_STYLE),
                        dcc.RadioItems(id='app3-condition-radio', options=[{'label': ' Anxiety', 'value': 'anxiety'}, {'label': ' Depression', 'value': 'depression'}], value='anxiety', inline=True, style={'marginBottom': '20px', 'marginTop': '8px'}, className='radio-items'),
                        html.Button('Generate Prediction', id='app3-predict-button', n_clicks=0, style=APP3_BUTTON_STYLE)
                    ])
                ], style=APP3_CARD_STYLE),
                dbc.Card([
                    dbc.CardHeader(html.H2("About", style=APP3_HEADER_STYLE)),
                    dbc.CardBody([
                        html.P("Uses an XGBoost model to predict mental health scores based on historical data."),
                        html.P("Adjust parameters to see predictions. Gauge shows score, chart shows monthly forecast.")
                    ])
                ], style=app3_info_card_style)
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '1%'}),
            # Right content area
            html.Div([
                dbc.Card([
                    dbc.CardHeader(html.H2("Prediction Results", style=APP3_HEADER_STYLE)),
                    dbc.CardBody([
                        html.Div(id='app3-prediction-output', style=app3_prediction_output_style),
                        dcc.Graph(id='app3-prediction-gauge', config={'displayModeBar': False})
                    ])
                ], style=APP3_CARD_STYLE),
                dbc.Card([
                    dbc.CardHeader(html.H2("Monthly Forecast", style=APP3_HEADER_STYLE)),
                    dbc.CardBody(dcc.Graph(id='app3-monthly-forecast', config={'displayModeBar': True, 'displaylogo': False}))
                ], style=APP3_CARD_STYLE),
                 dbc.Card([ # Moved Regional Comparison here
                    dbc.CardHeader(html.H2("Regional Comparison", style=APP3_HEADER_STYLE)),
                    dbc.CardBody(dcc.Graph(id='app3-regional-comparison', config={'displayModeBar': True, 'displaylogo': False}))
                ], style=APP3_CARD_STYLE),
            ], style={'width': '68%', 'display': 'inline-block', 'marginLeft': '1%', 'verticalAlign': 'top'}),
        ])
    ], style={'fontFamily': 'Roboto, sans-serif', 'margin': '0 auto', 'maxWidth': '1400px', 'padding': '0 20px 40px 20px', 'backgroundColor': APP3_COLORS['background'], 'color': APP3_COLORS['text']})


# --- Initialize App and Load Data/Models ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True,
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server
app.title = "Combined Mental Health Dashboard"

# Load data/models
app1_data_processor = App1DataProcessor()
app2_df_mainland = app2_load_data()
app3_model = app3_load_model()

# --- Main App Layout ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("Mental Health Dashboards", href="/", className="ms-2"),
            dbc.Nav([
                dbc.NavItem(dbc.NavLink("Trend Analysis (App1)", href="/app1")),
                dbc.NavItem(dbc.NavLink("Time Animation (App2)", href="/app2")),
                dbc.NavItem(dbc.NavLink("Predictor (App3)", href="/app3")),
            ], className="ms-auto", navbar=True)
        ]),
        color="dark",
        dark=True,
        sticky="top",
    ),
    html.Div(id='page-content', className="mt-4") # Content will be loaded here
])

# --- Callback Registration Functions ---

def register_app1_callbacks(app, data_processor):
    if data_processor.df is None: return # Don't register if data failed to load

    @app.callback(Output('app1-choropleth-map', 'figure'), Input('app1-indicator-dropdown', 'value'))
    def update_app1_choropleth(selected_indicator):
        indicator_name = selected_indicator
        for category, config in APP1_CATEGORIES.items():
            if category == selected_indicator:
                indicator_name = config['display_name']; break
        fig = px.choropleth(
            data_processor.df, locations='state_code', locationmode='USA-states', color=selected_indicator,
            scope='usa', color_continuous_scale=APP1_COLOR_SCALE, range_color=[0, 100], hover_name='geoName',
            hover_data=[selected_indicator, 'region'], labels={selected_indicator: 'Search Trend Index'},
            title=f'Depression Search Trends: {indicator_name} by State'
        )
        fig.update_layout(
            geo=dict(showlakes=True, lakecolor=APP1_MAP_COLORS['lake'], landcolor=APP1_MAP_COLORS['land'], showsubunits=True, subunitcolor=APP1_MAP_COLORS['subunit']),
            coloraxis_colorbar=dict(title='Search<br>Trend<br>Index', thicknessmode='pixels', thickness=20, lenmode='pixels', len=300),
            title={'font': {'size': 18, 'color': APP1_COLORS['primary']}, 'y': 0.95}, margin=dict(l=10, r=10, t=50, b=10), height=550,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Roboto, sans-serif', 'color': APP1_COLORS['text']}
        )
        return fig

    @app.callback(Output('app1-categories-graph', 'figure'), Input('app1-indicator-dropdown', 'value'))
    def update_app1_categories(_):
        fig = make_subplots(rows=2, cols=2, specs=[[{'type': 'choropleth'}] * 2] * 2, subplot_titles=[config['display_name'] for _, config in APP1_CATEGORIES.items()], horizontal_spacing=0.02, vertical_spacing=0.05)
        for i, (category, config) in enumerate(APP1_CATEGORIES.items()):
            row, col = divmod(i, 2); row += 1; col += 1
            geo_id = 'geo' if i == 0 else f'geo{i + 1}'
            fig.add_trace(go.Choropleth(locations=data_processor.df['state_code'], z=data_processor.df[category], locationmode='USA-states', colorscale=APP1_COLOR_SCALE, zmin=0, zmax=100, showscale=False, hovertemplate=f'<b>%{{location}}</b><br>{config["display_name"]}: %{{z}}<extra></extra>', geo=geo_id), row=row, col=col)
            fig.update_geos(scope='usa', showlakes=True, lakecolor=APP1_MAP_COLORS['lake'], showland=True, landcolor=APP1_MAP_COLORS['land'], showsubunits=True, subunitcolor=APP1_MAP_COLORS['subunit'], projection_scale=0.95, row=row, col=col)
        fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Roboto, sans-serif', 'color': APP1_COLORS['text']})
        for i in fig['layout']['annotations']: i['font'] = dict(size=14, color=APP1_COLORS['primary'], family='Roboto, sans-serif')
        return fig

    @app.callback(Output('app1-region-bar', 'figure'), Input('app1-indicator-dropdown', 'value'))
    def update_app1_region_bar(_):
        fig = px.bar(data_processor.region_stats, x='region', y='depression_index', color='depression_index', color_continuous_scale=APP1_COLOR_SCALE_REVERSE, title='Average Depression Search Trends by US Region', labels={'depression_index': 'Depression Search Index', 'region': 'Region'})
        fig.update_layout(xaxis_title='US Region', yaxis_title='Avg Search Index', coloraxis_showscale=False, title={'font': {'size': 16, 'color': APP1_COLORS['primary'], 'family': 'Roboto, sans-serif'}, 'y': 0.95}, height=220, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Roboto, sans-serif', 'color': APP1_COLORS['text']}, xaxis={'gridcolor': '#EEEEEE'}, yaxis={'gridcolor': '#EEEEEE'})
        for i, row in enumerate(data_processor.region_stats.itertuples()): fig.add_annotation(x=row.region, y=row.depression_index + 2, text=f"{row.depression_index:.1f}", showarrow=False, font=dict(family="Roboto, sans-serif", size=12, color=APP1_COLORS['primary']))
        return fig

    @app.callback(Output('app1-state-comparison-graph', 'figure'), [Input('app1-state1-dropdown', 'value'), Input('app1-state2-dropdown', 'value')])
    def update_app1_state_comparison(state1, state2):
        s1 = data_processor.df[data_processor.df['geoName'] == state1].iloc[0]
        s2 = data_processor.df[data_processor.df['geoName'] == state2].iloc[0]
        cats = ['depression_index'] + list(APP1_CATEGORIES.keys())
        comp_data = {'Category': ['Overall Index'] + [APP1_CATEGORIES[c]['display_name'] for c in cats[1:]], state1: [s1[c] for c in cats], state2: [s2[c] for c in cats]}
        comp = pd.DataFrame(comp_data)
        fig = px.bar(comp, x='Category', y=[state1, state2], barmode='group', title=f'Comparison: {state1} vs {state2}', labels={'value': 'Search Index', 'variable': 'State'}, color_discrete_sequence=[APP1_COLORS['accent'], APP1_COLORS['secondary']])
        fig.update_layout(xaxis_title='Search Category', yaxis_title='Search Index', yaxis=dict(range=[0, 100]), legend_title_text='State', title={'font': {'size': 18, 'color': APP1_COLORS['primary'], 'family': 'Roboto, sans-serif'}, 'y': 0.95}, height=400, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family': 'Roboto, sans-serif', 'color': APP1_COLORS['text']}, xaxis={'gridcolor': '#EEEEEE'}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        # Add annotations (optional, can make plot busy)
        # for i, cat in enumerate(comp['Category']):
        #     fig.add_annotation(x=i - 0.2, y=comp[state1].iloc[i] + 3, text=f"{comp[state1].iloc[i]:.1f}", showarrow=False, font=dict(family="Roboto, sans-serif", size=10, color=APP1_COLORS['accent']))
        #     fig.add_annotation(x=i + 0.2, y=comp[state2].iloc[i] + 3, text=f"{comp[state2].iloc[i]:.1f}", showarrow=False, font=dict(family="Roboto, sans-serif", size=10, color=APP1_COLORS['secondary']))
        return fig

def register_app2_callbacks(app, df_mainland):
    if df_mainland.empty: return # Don't register if data failed to load

    @app.callback(
        Output('app2-animated-map', 'figure'),
        [Input('app2-generate-button', 'n_clicks')],
        [State('app2-indicator-selector', 'value'),
         State('app2-colorscale-selector', 'value'),
         State('app2-speed-slider', 'value')]
    )
    def update_app2_animated_map(n_clicks, indicator, colorscale, speed):
        # Use clientside callback for initial load if preferred, or just generate on first load
        # if n_clicks is None and not dash.ctx.triggered_id: # Prevent initial auto-generation if desired
        #     return go.Figure()
        fig = app2_create_animated_choropleth(df_mainland, indicator=indicator, colorscale=colorscale, frame_duration=speed)
        return fig

    @app.callback(Output('app2-top-states-chart', 'figure'), [Input('app2-indicator-selector', 'value')])
    def update_app2_top_states_chart(indicator):
        fig = app2_create_top_states_chart(df_mainland, indicator=indicator)
        return fig

    @app.callback(Output('app2-map-title', 'children'), [Input('app2-indicator-selector', 'value')])
    def update_app2_map_title(indicator):
        if indicator == 'combined': return "Animated Map: Depression & Anxiety Search Interest"
        else: return f"Animated Map: {indicator} Search Interest"

    # Optional: Clientside callback to trigger button click on initial load for App2 map
    clientside_callback(
        """
        function(pathname) {
            if (pathname === '/app2') {
                // Small delay to ensure layout is rendered
                setTimeout(function() {
                    const button = document.getElementById('app2-generate-button');
                    if (button) {
                        button.click();
                    }
                }, 500);
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('app2-generate-button', 'data-dummy-output'), # Dummy output
        Input('url', 'pathname')
    )


def register_app3_callbacks(app, model, regions_list, months_dict):
    @app.callback(
        [Output('app3-prediction-output', 'children'),
         Output('app3-prediction-gauge', 'figure'),
         Output('app3-monthly-forecast', 'figure'),
         Output('app3-regional-comparison', 'figure')],
        [Input('app3-predict-button', 'n_clicks')],
        [State('app3-year-slider', 'value'),
         State('app3-month-dropdown', 'value'),
         State('app3-region-dropdown', 'value'),
         State('app3-condition-radio', 'value')]
    )
    def update_app3_prediction(n_clicks, year, month, region, condition):
        if n_clicks == 0:
            return ("Click 'Generate Prediction' to see results", app3_create_empty_gauge(), app3_create_empty_forecast(), app3_create_empty_comparison())

        is_anxiety = condition == 'anxiety'
        condition_text = "Anxiety" if is_anxiety else "Depression"
        month_text = months_dict[month]

        try:
            features = app3_create_example_data(year, month, is_anxiety, region, model, regions_list)
            prediction = model.predict(features)[0]

            prediction_text = html.Div([
                html.Span("Predicted Score: ", style={'fontWeight': 'bold'}),
                html.Span(f"{prediction:.2f}", style={'fontSize': '24px', 'color': APP3_COLORS['secondary']}), html.Br(),
                html.Span(f"For {condition_text} in {region} during {month_text} {year}")
            ])
            gauge_fig = app3_create_gauge_chart(prediction, condition_text)
            forecast_fig = app3_create_monthly_forecast(year, region, condition_text, is_anxiety, model, regions_list, months_dict)
            comparison_fig = app3_create_regional_comparison(year, month, is_anxiety, condition_text, model, regions_list, months_dict)

            return prediction_text, gauge_fig, forecast_fig, comparison_fig

        except Exception as e:
            print(f"Error during App3 prediction/plotting: {e}")
            error_message = dbc.Alert(f"An error occurred: {e}", color="danger")
            return error_message, app3_create_empty_gauge(), app3_create_empty_forecast(), app3_create_empty_comparison()

# --- Register Callbacks ---
register_app1_callbacks(app, app1_data_processor)
register_app2_callbacks(app, app2_df_mainland)
register_app3_callbacks(app, app3_model, APP3_REGIONS, APP3_MONTHS)

# --- Navigation Callback ---
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/app1':
        return create_app1_layout(app1_data_processor)
    elif pathname == '/app2':
        return create_app2_layout(app2_df_mainland)
    elif pathname == '/app3':
        return create_app3_layout()
    else: # Default to App1 or an index page
        return create_app1_layout(app1_data_processor) # Or create an index page

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
