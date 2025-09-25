import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import calendar
from pathlib import Path


@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    if 'last_updated' in data.columns:
        data['last_updated'] = pd.to_datetime(data['last_updated'], format='%Y-%m-%d %H:%M', errors='coerce')
        data['year'] = data['last_updated'].dt.year
        data['month'] = data['last_updated'].dt.month_name()
        data['day'] = data['last_updated'].dt.day
    drop_cols = ['last_updated_epoch','temperature_fahrenheit','feels_like_fahrenheit','moonrise','moonset','moon_phase','moon_illumination']
    data = data.drop(columns=drop_cols, errors='ignore')
    return data


def css():
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f1115; }
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
        h1,h2,h3,h4,h5,h6,p, .stMetric { color:#e5e7eb !important; }
        .card { background:#111827; border:1px solid #1f2937; border-radius:16px; padding:16px; }
        .muted { color:#9ca3af; }
        .hero-temp { font-size:48px; font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header_metrics(row: pd.Series) -> None:
    c1, c2, c3 = st.columns([1.2, 0.8, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### {row.get('location_name','Unknown')}")
        st.markdown(f"<span class='muted'>{row.get('country','')}</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='hero-temp'>{float(row['temperature_celsius']):.0f}° C</div>", unsafe_allow_html=True)
        st.markdown(f"{row.get('condition_text','')}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric('Humidity', f"{int(row.get('humidity',0))}%")
        st.metric('Wind', f"{float(row.get('wind_kph',0)):0.1f} kph")
        st.metric('UV', f"{float(row.get('uv_index',0)):0.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if 'sunrise' in row and 'sunset' in row:
            st.metric('Sunrise', row.get('sunrise','—'))
            st.metric('Sunset', row.get('sunset','—'))
        st.markdown('</div>', unsafe_allow_html=True)


def tab_temperature(data: pd.DataFrame):
    scope = st.selectbox('Scope', ['Global','Country','Location'], key='t_scope_app')
    group_key = {'Global': None, 'Country': 'country', 'Location': 'location_name'}[scope]

    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(data['temperature_celsius'], bins=40, kde=True, color='#fbbf24', ax=ax)
    ax.set_xlabel('Temperature (°C)')
    st.pyplot(fig)

    if 'last_updated' in data.columns and not data['last_updated'].isna().all():
        ts = data.sort_values('last_updated')
        st.plotly_chart(px.line(ts, x='last_updated', y='temperature_celsius', color=group_key), use_container_width=True)

    if 'feels_like_celsius' in data.columns:
        st.plotly_chart(px.scatter(data, x='temperature_celsius', y='feels_like_celsius', color=group_key or 'country', opacity=0.6), use_container_width=True)


def tab_wind_pressure(data: pd.DataFrame):
    if 'wind_direction' in data.columns:
        dir_mean = data.groupby('wind_direction')['wind_kph'].mean().reset_index()
        st.plotly_chart(px.bar_polar(dir_mean, r='wind_kph', theta='wind_direction', color='wind_direction', template='plotly_dark'), use_container_width=True)
    if 'pressure_mb' in data.columns:
        st.plotly_chart(px.scatter(data, x='pressure_mb', y='temperature_celsius', color='country', opacity=0.6), use_container_width=True)


def tab_precip_hum_clouds(data: pd.DataFrame):
    if 'precip_mm' in data.columns:
        precip_country = data.groupby('country')['precip_mm'].mean().sort_values(ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(8,4))
        precip_country.plot(kind='bar', color='#60a5fa', ax=ax)
        ax.set_ylabel('Avg precip (mm)')
        st.pyplot(fig)
    if 'condition_text' in data.columns and 'visibility_km' in data.columns:
        st.plotly_chart(px.box(data, x='condition_text', y='visibility_km'), use_container_width=True)


def tab_air_quality(data: pd.DataFrame):
    pollutant_cols = [
        'air_quality_PM2.5','air_quality_PM10','air_quality_Ozone',
        'air_quality_Nitrogen_dioxide','air_quality_Sulphur_dioxide',
        'air_quality_Carbon_Monoxide'
    ]
    present = [c for c in pollutant_cols if c in data.columns]
    if present:
        loc_means = data.groupby('location_name')[present].mean().reset_index()
        melted = loc_means.melt(id_vars='location_name', value_vars=present,
                                 var_name='pollutant', value_name='value')
        st.plotly_chart(
            px.bar(melted, x='location_name', y='value', color='pollutant', barmode='stack'),
            use_container_width=True,
        )
        corr_cols = ['temperature_celsius','humidity','uv_index','wind_kph','pressure_mb'] + present
        corr_cols = [c for c in corr_cols if c in data.columns]
        if corr_cols:
            corr = data[corr_cols].corr(numeric_only=True)
            st.plotly_chart(
                px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r'),
                use_container_width=True,
            )
    if 'air_quality_us-epa-index' in data.columns and 'condition_text' in data.columns:
        st.plotly_chart(px.box(data, x='condition_text', y='air_quality_us-epa-index'), use_container_width=True)


def tab_mapping(data: pd.DataFrame):
    country_temp = (
        data.groupby('country', dropna=True)['temperature_celsius']
        .mean()
        .reset_index()
    )
    if not country_temp.empty:
        fig_ch = px.choropleth(
            country_temp,
            locations='country',
            locationmode='country names',
            color='temperature_celsius',
            color_continuous_scale='RdYlBu_r',
            range_color=[-20,50],
            labels={'temperature_celsius':'Avg Temp (°C)'}
        )
        fig_ch.update_geos(showcountries=True)
        fig_ch.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_ch, use_container_width=True)

def build_temperature_gauge(data: pd.DataFrame) -> None:
    sorted_df = data.sort_values('temperature_celsius').reset_index(drop=True)
    if sorted_df.empty:
        return
    temp_min = float(sorted_df['temperature_celsius'].min())
    temp_max = float(sorted_df['temperature_celsius'].max())
    initial_temp = float(sorted_df['temperature_celsius'].iloc[0])

    bins = 5
    bin_edges = np.linspace(temp_min, temp_max, bins + 1)
    colors = ["#2b83ba", "#7fcdbb", "#ffffbf", "#fdae61", "#d7191c"]
    steps = []
    for i in range(bins):
        steps.append({'range': [float(bin_edges[i]), float(bin_edges[i + 1])], 'color': colors[i]})

    # Use session value if available so we can place the slider below the chart
    sel_temp = float(st.session_state.get('gauge_slider', initial_temp))
    closest_idx = (sorted_df['temperature_celsius'] - sel_temp).abs().idxmin()
    row = sorted_df.loc[closest_idx]

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=float(row['temperature_celsius']),
        title={'text': 'Temperature (Celsius)'},
        gauge={'axis': {'range': [temp_min, temp_max]}, 'bar': {'color': 'darkblue'}, 'steps': steps}
    ))
    fig.update_layout(height=300)
    # Show contextual details beneath the gauge
    country = row['country'] if 'country' in row.index else 'Unknown'
    location_name = row['location_name'] if 'location_name' in row.index else 'Unknown'
    date_time = row['last_updated'] if 'last_updated' in row.index else 'Unknown'
    try:
        date_time = pd.to_datetime(date_time).strftime('%Y-%m-%d %H:%M') if pd.notna(date_time) else 'Unknown'
    except Exception:
        pass
    st.caption(f"Country: {country} | Location: {location_name} | Date & Time: {date_time}")
    st.plotly_chart(fig, use_container_width=True)
    # Slider BELOW the chart
    st.slider('Temp (°C)', min_value=temp_min, max_value=temp_max, value=sel_temp, step=0.1, key='gauge_slider')


def main():
    st.set_page_config(page_title='Climate Dashboard', layout='wide')
    css()
    st.title('Climate Dashboard')

    csv_path = 'global_weather.csv'
    if not Path(csv_path).exists():
        st.error('CSV not found')
        return
    data = load_data(csv_path)

    # Filters
    st.sidebar.header('Filters')
    countries = ['All'] + sorted(data['country'].dropna().unique().tolist())
    sel_country = st.sidebar.selectbox('Country', countries)
    filtered = data if sel_country == 'All' else data[data['country'] == sel_country]

    # Current row
    if 'last_updated' in filtered.columns and not filtered['last_updated'].isna().all():
        current = filtered.sort_values('last_updated').iloc[-1]
    else:
        current = filtered.iloc[0]

    # Header metrics; gauge and mapping side-by-side
    header_metrics(current)
    c1, c2 = st.columns([0.7, 1.3])
    with c1:
        st.subheader('Temperature Gauge')
        build_temperature_gauge(filtered)
    with c2:
        st.subheader('Choropleth: Avg Temperature by Country (°C')
        tab_mapping(filtered)


    t1, t2, t3, t4 = st.tabs(['Temperature & Feels','Wind & Pressure','Precip • Visibility','Air Quality'])
    with t1:
        tab_temperature(filtered)
    with t2:
        tab_wind_pressure(filtered)
    with t3:
        tab_precip_hum_clouds(filtered)
    with t4:
        tab_air_quality(filtered)



if __name__ == '__main__':
    main()


