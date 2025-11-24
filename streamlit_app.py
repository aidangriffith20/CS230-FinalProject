"""
Name:       Aidan J. Griffith
CS230:      Section XXX
Data:       Alternative Fuel Stations in MA
URL:        (add Streamlit Cloud URL if you deploy)

Description:
This program is an interactive data explorer for Alternative Fuel Stations in MA.
Users can:
- Explore how many stations of each fuel type exist in a selected city.
- Find stations within a given radius of a selected city and visualize them on a map.
- Use an hour-of-day slider to see which stations are likely to be open, based on
  the 'Access Days Time' field, using categorized opening hours.

References:
https://www.youtube.com/watch?v=8G4cD7ofgCM
https://www.youtube.com/watch?v=IzBk58Eorr4


AI Usage:
1) Initial spacing, and commenting to increase readibility of my code
    Include the following notations in the comment when I use the corresponding coding techniques
2) Small Troubleshooting when stuck on a task
    - Used to fully grasp regex, at both a conceptual level, and its application here
3)See AI declaration 1
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import re as re
import plotly as px

DATA_FILE = "alt_fuel_stations.csv"


# Data Loading & Helpers

@st.cache_data
# Used to save slider from consistently reloading data

def load_data(file_path):
    """
    Load the dataset and perform initial cleaning
    """
    df = pd.read_csv(file_path)

    # Keep only rows with lat/long
    df = df.dropna(subset=["Latitude", "Longitude"])

    # Calls to the function that analyzes Access Days Time
    # (provided hours information in CSV) and creates the three columns
    df = add_hours_categories(df)

    # Create a sorted list of cities using a list comprehension #[LISTCOMP]
    cities = sorted([c for c in df["City"].dropna().unique()])

    # Unique fuel types
    fuel_types = sorted(df["Fuel Type Code"].dropna().unique())

    # [FUNCRETURN2] Return multiple values from helper function
    return df, cities, fuel_types



def add_hours_categories(df):
    """
 Create our new columns in the data frame, and add the corresponding values
 from parse_access_days_time
    """

    # Create columns for our use #[COLUMNS]
    df["hours_category"] = np.nan
    df["open_hour"] = np.nan
    df["close_hour"] = np.nan

    # [ITERLOOP] Iterate through rows to set categories
    for idx, value in df["Access Days Time"].items():

# Sending hours into parse_access_days_time to properly assign values to rows
        category, open_h, close_h = parse_access_days_time(value)

        df.at[idx, "hours_category"] = category
        df.at[idx, "open_hour"] = open_h
        df.at[idx, "close_hour"] = close_h

    return df




def parse_access_days_time(value):
    """
    Categorize station open hours using simple logic:
      - 24_7 → always open
      - business_hours → approx 9–17
      - parseable_hours → regex-detected times like '6am-6pm'
      - unknown_restricted → anything else
    """

    text = str(value).lower()

    # Category 1: 24/7
    if "24 hours" in text or "24hrs" in text or "24 hr" in text:
        return "24_7", 0, 24

    # Category 2: Business Hours
    if "business hours" in text or "dealership business hours" in text or "garage business hours" in text:
        return "business_hours", 9, 17

    # Category 3: Regex Parsing for specific times
    match = TIME_REGEX.search(text)
    if match:
        start_raw, end_raw = match.groups()
        start_hour = parse_time_to_24h(start_raw)
        end_hour = parse_time_to_24h(end_raw)

        if end_hour > start_hour:
            return "parseable_hours", start_hour, end_hour

    # Category 4: Unknown / Restricted
    return "unknown_restricted", None, None

#REGEX use to capture varying opening hours.
# See Youtube video 2, and AI Declaration 2
TIME_REGEX = re.compile(
    r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*[-–]\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm))',
    re.IGNORECASE

    #Parenthesis create 'groups' that we can assign below
)

def parse_time_to_24h(time):
    """
    Convert times like '6am', '6:30pm', '10 pm' into 24-hour integer hours.
    To
        ignore minutes to keep slider logic simple (going to always round down).
        Make a 24 hour time scale for easier slider
    """
    t = time.strip().lower().replace(" ", "")  # normalize: remove spaces, lowercase

    # Check for and remove minutes
    if ":" in t:
        hour_str = t.split(":")[0]
    else:
        hour_str = t[:-2]  # cuts off am/pm

    hour = int(hour_str)

    # Apply am/pm rules

    if "pm" in t and hour != 12:
        hour += 12
    if "am" in t and hour == 12:
        hour = 0

    return hour



# [FUNC2P] Function with 2+ params (one with default)
def filter_by_city_and_fuel(df, city, fuel):
    """
    Filter stations by city and optionally by fuel type.
    """
    mask = df["City"] == city

    if fuel:
        mask &= df["Fuel Type Code"] == fuel   # #[FILTER2] Filter by two conditions

    filtered = df[mask].copy()

    # Sort by station name #[SORT]
    filtered = filtered.sort_values(by="Station Name")

    return filtered


from haversine import haversine, Unit

def distance_miles(lat1, lon1, lat2, lon2):
    return haversine((lat1, lon1), (lat2, lon2), unit=Unit.MILES)


def is_station_open(row, hour):
    """
    Determine if a station is open at a given hour based on hours_category
    and approximate open/close hours.
    """
    cat = row.get("hours_category", "unknown_restricted")

    if cat == "24_7":
        return True

    if cat == "business_hours":
        open_h = row.get("open_hour", 9)
        close_h = row.get("close_hour", 17)
        return open_h <= hour < close_h

    if cat == "parseable_hours":
        open_h = row.get("open_hour")
        close_h = row.get("close_hour")

        if open_h is not None and close_h is not None:
            return open_h <= hour < close_h


    # unknown / restricted
    return False


# Visualization Helpers

#Declaration 4
def create_pydeck_map(df):
    """
    Generic PyDeck scatterplot map for a given dataframe of stations.
    """

    # Compute map center
    mid_lat = df["Latitude"].mean()
    mid_lon = df["Longitude"].mean()

    # Default color (blue)
    default_color = [0, 100, 200]

    # If a 'color' column exists, use it, otherwise constant color
    has_color = "color" in df.columns

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[Longitude, Latitude]",
        get_radius=300,
        get_fill_color="color" if has_color else default_color,
        pickable=True,
    )

# info available when hovering

    tooltip = {
        "html": """
        <b>{Station Name}</b><br/>
        Fuel Type: {Fuel Type Code}<br/>
        City: {City}, {State}<br/>
        Access Hours: {Access Days Time}
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }

    view_state = pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=9, pitch=0)

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
    )

    return deck


#declaration 5
# Streamlit Pages


def page_overview(df):
  # Home page for Streamlit

    st.title("MA Alternative Fuel Stations Explorer")
    st.markdown(
        """
        Welcome! Use the sidebar to explore:
        - Counts of stations by fuel type and city
        - Stations within a radius of a selected city
        - Stations likely to be open at a given hour of day
        """
    )

    st.subheader("Basic Summary Statistics")

    # Number of stations per fuel type
    fuel_counts = df["Fuel Type Code"].value_counts().reset_index()
    fuel_counts.columns = ["Fuel Type", "Count"]

    # [CHART1] Example bar chart
    st.bar_chart(fuel_counts.set_index("Fuel Type"))

    st.write("Raw data preview:")
    st.dataframe(df.head())


def page_fuel_by_city(df, cities, fuel_types):
        st.header("Stations by Fuel Type and City")

        # Sidebar / main controls
        # [ST1] Dropdowns for city & fuel type
        city = st.selectbox("Select City", options=cities, index=29)

        fuel = st.selectbox(
            "Select Fuel Type (optional)",
            options=["(All)"] + fuel_types,
            index=0,
        )

        fuel_filter = None if fuel == "(All)" else fuel

        # Use helper filter function #[FUNCCALL2]
        subset = filter_by_city_and_fuel(df, city, fuel_filter)

        st.write(f"Found **{len(subset)}** stations in **{city}**.")

        # Pivot table: counts by fuel type #[PIVOTTABLE]
        pivot = (
            df[df["City"] == city]
            .pivot_table(
                values="ID",
                index="Fuel Type Code",
                aggfunc="count"
            )
            .rename(columns={"ID": "Station Count"})
            .sort_values("Station Count", ascending=False)
        )
        st.subheader("Counts by Fuel Type")
        st.dataframe(pivot)


#declaration 3
def page_radius_search(df, cities):
    st.header("Stations Within Radius of a City")

    # [ST1] City dropdown
    city = st.selectbox("Select City for Center Point", options=cities, key="radius_city", index =29)

    # [ST2] Slider for radius
    radius_miles = st.slider(
        "Radius (miles)",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )

    # Center point for city
    city_df = df[df["City"] == city]
    if city_df.empty:
        st.warning("No stations with coordinates for this city.")
        return

    center_lat = city_df["Latitude"].mean()
    center_lon = city_df["Longitude"].mean()

    # Compute distance to each station
    # [FILTER1]

    df["distance_miles"] = df.apply(
        lambda row: distance_miles(center_lat, center_lon, row["Latitude"], row["Longitude"]),
        axis=1,
    )

    within = df[df["distance_miles"] <= radius_miles].copy()  # #[FILTER2] distance + maybe other later

    st.write(f"Found **{len(within)}** stations within **{radius_miles} miles** of **{city}**.")

    if within.empty:
        st.info("No stations found within this radius.")
        return


    # PyDeck map
    deck = create_pydeck_map(within)
    if deck:
        st.subheader("Map of Stations Within Radius")
        st.pydeck_chart(deck)

def page_hour_slider(df):
    st.header("Stations Open by Hour of Day")

    st.markdown(
        """
        This view uses a simplified interpretation of **Access Days Time** to
        categorize stations as:
        - 24/7
        - Business hours (approx 9–17)
        - Unknown / restricted (shown as closed)
        """
    )
#declaration 3
    # [ST2] Slider for hour of day

    st.markdown("""
    ### Legend
    - <span style='color: rgb(0,200,0); font-size: 24px;'>●</span> Open  
    - <span style='color: rgb(200,0,0); font-size: 24px;'>●</span> Closed  
    - <span style='color: gray; font-size: 24px;'>●</span> Unknown/Restricted  
    """, unsafe_allow_html=True)

    hour = st.slider("Select Hour of Day (0–23)", min_value=0, max_value=23, value=9)


    # Compute open/closed boolean
    # [DICTMETHOD] Use .get on the row (also used in is_station_open)

    df["is_open"] = df.apply(lambda row: is_station_open(row, hour), axis=1)

    # Map colors based on category & open status
    color_map = {
        "open": [0, 200, 0],
        "closed": [200, 0, 0],
        "unknown_restricted": [128, 128, 128],
    }

    def assign_color(row):
        if row["hours_category"] in ["unknown_restricted"]:
            return color_map["unknown_restricted"]

        return color_map["open"] if row["is_open"] else color_map["closed"]

#applying to every row
    df["color"] = df.apply(assign_color, axis=1)

#Open stations
    open_count = df["is_open"].sum()
    st.write(f"At hour **{hour}:00**, approximately **{open_count}** stations are open.")

    # Map
    deck = create_pydeck_map(df)
    if deck:
        st.subheader("Map Colored by Open/Closed Status")
        st.pydeck_chart(deck)


#Pie Chart

    st.subheader("Station Availability Breakdown at This Hour")

    # Count groups
    total_stations = len(df)
    open_count = df["is_open"].sum()
    unknown_count = len(df[df["hours_category"] == "unknown_restricted"])
    closed_count = total_stations - open_count - unknown_count

    availability_counts = pd.DataFrame({
        "Status": ["Open", "Closed", "Unknown/Restricted"],
        "Count": [open_count, closed_count, unknown_count]
    })

    # Plot pie chart
    fig = px.pie(
        availability_counts,
        values="Count",
        names="Status",
        title="Station Availability",
        color="Status",
        color_discrete_map={
            "Open": "green",
            "Closed": "red",
            "Unknown/Restricted": "gray"
        }
    )

    st.plotly_chart(fig)


# Main App


def main():
    st.set_page_config(page_title="Alternative Fuel Stations Explorer", layout="wide")

    # [ST3] Sidebar for navigation and layout
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        ["Overview", "Fuel Type by City", "Radius Search", "Open by Hour"],
    )

    # Load data once
    file_exists = Path(DATA_FILE).exists()
    if not file_exists:
        st.error(f"Data file '{DATA_FILE}' not found. Please place the CSV in the same folder as this file.")
        return

    df, cities, fuel_types = load_data(DATA_FILE)

    if page == "Overview":
        page_overview(df)
    elif page == "Fuel Type by City":
        page_fuel_by_city(df, cities, fuel_types)
    elif page == "Radius Search":
        page_radius_search(df, cities)
    elif page == "Open by Hour":
        page_hour_slider(df)


if __name__ == "__main__":
    main()
