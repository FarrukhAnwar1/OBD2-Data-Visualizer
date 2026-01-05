import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Set page layout to wide for better graphing view
st.set_page_config(layout="wide", page_title="OBD2 Data Visualizer")

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    /* 1. Sidebar sizing */
    [data-testid="stSidebar"] {
        min-width: 450px;
        max-width: 600px;
    }

    /* 2. AGGRESSIVE WHITESPACE REMOVAL */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0rem !important;
    }
    [data-testid="stSidebar"] h2 {
        margin-top: 0px !important;
        padding-top: 10px !important;
        margin-bottom: 10px !important;
    }

    /* 3. Style the Scrollable Container (Sensor List) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #444;
        background-color: #262730; 
    }

    /* 4. Custom Tag Style for Selected List */
    .selected-tag {
        display: inline-block;
        background-color: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
        font-weight: 500;
    }

    /* 5. NEW: Container for Selected Tags (Limit to ~3 rows) */
    .selected-container {
        min-height: 70px;
        max-height: 70px;      /* Approx height for 3 rows */
        overflow-y: auto;       /* Scroll vertically if content exceeds height */
        padding: 5px;
        border: 1px solid #444; /* Subtle border to define the area */
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.05); /* Slight background contrast */
    }

    /* 6. Tighter horizontal rules */
    hr {
        margin-top: 5px !important;
        margin-bottom: 5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("OBD2 Data Visualizer")

# --- 1. File Upload Section ---
uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"], label_visibility="collapsed")


# --- DATA LOADING ---
@st.cache_data
def load_data(file):
    # Read CSV with semi-colon separator
    df = pd.read_csv(file, sep=';', on_bad_lines='skip')
    df.columns = df.columns.str.strip()

    # FORMAT A: Original format (SECONDS, PID, VALUE)
    if all(col in df.columns for col in ['SECONDS', 'PID', 'VALUE']):
        df_pivot_load = df.pivot_table(index='SECONDS', columns='PID', values='VALUE', aggfunc='mean')
        df_pivot_load = df_pivot_load.sort_index()
        df_pivot_load = df_pivot_load.ffill().bfill()
        return df_pivot_load

    # FORMAT B: New format (time(ms) and sensor columns)
    elif 'time(ms)' in df.columns:
        # Convert time from ms to seconds
        df['SECONDS'] = df['time(ms)'] / 1000.0
        df = df.drop(columns=['time(ms)'])

        # Replace the dash '-' representing null values with actual NaNs
        df = df.replace('-', np.nan)

        # Identify sensor columns (exclude the new SECONDS column)
        sensor_cols = [c for c in df.columns if c != 'SECONDS']

        # Convert sensor columns to numeric (coercing strings like "Closed Loop" to NaN)
        for col in sensor_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Group by SECONDS in case of duplicate timestamps, taking the mean
        df_processed = df.groupby('SECONDS').mean().sort_index()

        # Drop columns that are completely empty after conversion
        df_processed = df_processed.dropna(axis=1, how='all')

        # Fill holes in the data
        df_processed = df_processed.ffill().bfill()
        return df_processed

    return None


# --- STATE MANAGEMENT HELPERS ---
def toggle_sensor(sensor_name):
    """Callback: Sync Widget -> Dictionary (User clicked a specific box)"""
    widget_key = f"chk_{sensor_name}"
    if widget_key in st.session_state:
        st.session_state['sensor_states_dict'][sensor_name] = st.session_state[widget_key]


def get_selected_list():
    """Return list of sensor names where value is True."""
    return [k for k, v in st.session_state['sensor_states_dict'].items() if v]


def select_visible_matches():
    """
    Callback: Sync Search Matches -> Dictionary AND Widget
    """
    search_txt = st.session_state.search_box.lower()
    all_keys = st.session_state['sensor_states_dict'].keys()

    if search_txt:
        matches = [k for k in all_keys if search_txt in k.lower()]
    else:
        matches = all_keys

    for m in matches:
        st.session_state['sensor_states_dict'][m] = True
        st.session_state[f"chk_{m}"] = True


def deselect_all():
    """
    Callback: Uncheck All -> Dictionary AND Widget
    """
    for k in st.session_state['sensor_states_dict']:
        st.session_state['sensor_states_dict'][k] = False
        st.session_state[f"chk_{k}"] = False


def clear_search():
    st.session_state.search_box = ""


# --- MAIN APP LOGIC ---

if uploaded_file is not None:
    df_pivot = load_data(uploaded_file)

    if df_pivot is None:
        st.error(
            "Error: The CSV file format is not recognized. Expected columns: (SECONDS; PID; VALUE) OR (time(ms); "
            "...sensors).")
    else:
        # --- INITIALIZATION ---
        all_metrics = df_pivot.columns.tolist()

        # Initialize the persistent dictionary if it doesn't exist
        if 'sensor_states_dict' not in st.session_state:
            st.session_state['sensor_states_dict'] = {sensor: False for sensor in all_metrics}
            # Initialize widget keys too
            for sensor in all_metrics:
                if f"chk_{sensor}" not in st.session_state:
                    st.session_state[f"chk_{sensor}"] = False
        else:
            # Sync any new columns if file changed
            for sensor in all_metrics:
                if sensor not in st.session_state['sensor_states_dict']:
                    st.session_state['sensor_states_dict'][sensor] = False

        # --- 2. Sidebar Controls ---
        st.sidebar.header("Graph Settings")

        # --- A. GRAPH CONTROLS (AT TOP) ---
        c1, c2 = st.sidebar.columns(2)
        with c1:
            normalize = st.checkbox("Normalize Data", value=False)
        with c2:
            split_graphs = st.checkbox("Split Graphs", value=False)

        trigger_plot = st.sidebar.button("Update Graph", type="primary", width='stretch')

        st.sidebar.markdown("---")

        # --- B. VISUAL CONFIRMATION (SCROLLABLE ~3 ROWS) ---
        current_selection = get_selected_list()

        if current_selection:
            st.sidebar.markdown(f"**Selected ({len(current_selection)}):**")
            tags_html = "".join([f"<span class='selected-tag'>{s}</span>" for s in current_selection])
            st.sidebar.markdown(f"<div class='selected-container'>{tags_html}</div>", unsafe_allow_html=True)
        else:
            st.sidebar.markdown("**Selected:** None")

        st.sidebar.markdown("---")

        # --- C. SEARCH ---
        search_text = st.sidebar.text_input(
            "🔍 Search Sensors:",
            value="",
            placeholder="Type to filter list...",
            key="search_box"
        )

        if search_text:
            filtered_options = [m for m in all_metrics if search_text.lower() in m.lower()]
            btn_label = f"Select Matches ({len(filtered_options)})"
        else:
            filtered_options = all_metrics
            btn_label = "Select All"

        # --- D. SCROLLABLE SENSOR LIST ---
        with st.sidebar.container(height=200, border=True):
            for sensor in filtered_options:
                key_name = f"chk_{sensor}"
                if key_name not in st.session_state:
                    st.session_state[key_name] = st.session_state['sensor_states_dict'][sensor]

                st.checkbox(
                    sensor,
                    key=key_name,
                    on_change=toggle_sensor,
                    args=(sensor,)
                )

        # --- E. ACTION BUTTONS ---
        c_act1, c_act2, c_act3 = st.sidebar.columns([1, 1, 0.8])
        with c_act1:
            st.button(btn_label, on_click=select_visible_matches, key='btn_sel_all', width='stretch')
        with c_act2:
            st.button("Uncheck All", on_click=deselect_all, key='btn_uncheck', width='stretch')
        with c_act3:
            st.button("Clear", on_click=clear_search, key='btn_clr_srch', width='stretch')

        # --- 3. Plotting Logic ---
        if trigger_plot or 'graph_active' in st.session_state:
            st.session_state['graph_active'] = True

            metrics_to_plot = get_selected_list()

            if metrics_to_plot:
                st.subheader(f"Timeline")

                valid_metrics = [m for m in metrics_to_plot if m in df_pivot.columns]

                if valid_metrics:
                    plot_data = df_pivot[valid_metrics].copy()

                    if normalize:
                        # Prevent division by zero if all values are the same
                        denom = (plot_data.max() - plot_data.min())
                        plot_data = (plot_data - plot_data.min()) / denom.replace(0, 1)

                    # --- PLOTTING ---
                    if split_graphs:
                        df_melted = plot_data.reset_index().melt(id_vars='SECONDS', var_name='Sensor',
                                                                 value_name='Value')

                        dynamic_height = max(600, 250 * len(valid_metrics))

                        fig = px.line(
                            df_melted,
                            x='SECONDS',
                            y='Value',
                            color='Sensor',
                            facet_row='Sensor',
                            height=dynamic_height
                        )
                        fig.update_yaxes(matches=None)
                        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                    else:
                        fig = px.line(plot_data, x=plot_data.index, y=valid_metrics, height=600)

                    fig.update_layout(
                        xaxis_title="Time (Seconds)",
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        ),
                    )

                    fig.update_xaxes(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across")
                    st.plotly_chart(fig, width='stretch')

                    # --- STATISTICS (Expanded by Default) ---
                    with st.expander("Show Statistics", expanded=True):
                        st.dataframe(plot_data.describe(), width='stretch')
                else:
                    st.warning("Selected sensors not found in data.")
            else:
                st.info("👈 Select sensors in the sidebar and click 'Update Graph'.")

else:
    st.info("👋 Upload a CSV file to get started.")
