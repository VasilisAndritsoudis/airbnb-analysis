import time
from collections import Counter
import pandas as pd
from scipy.stats import pointbiserialr

import pymongo
import streamlit as st
from annotated_text import annotated_text  # pip install st-annotated-text

import pydeck as pdk
import plotly.figure_factory as ff
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


# Login into MongoDB and return the collection
def load_mongodb_collection():
    # MongoDB info
    mongo_host = ""
    mongo_port = 0
    mongo_database = ""
    mongo_collection = ""
    mongo_username = ""
    mongo_password = ""

    # Connect to MongoDB
    client = pymongo.MongoClient(mongo_host, mongo_port, username=mongo_username, password=mongo_password)
    # Get the database
    db = client[mongo_database]
    # Get the collection
    collection = db[mongo_collection]

    return collection


# 1. Functions for calculating of correlation matrix for the type of property and price
# ==================================================================================
# Find the k clusters using k-Means for the property details
def cluster_properties(df, features, num_clusters):
    # Conducting the k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['property_type'] = kmeans.fit_predict(features)

    # One-hot encoding the cluster labels
    ohe = OneHotEncoder(sparse=False, dtype=int)
    property_type_ohe = ohe.fit_transform(df[['property_type']])

    # Creating a dataframe with the one-hot encoded columns
    type_labels = [f'Type {i + 1}' for i in range(num_clusters)]
    df_ohe = pd.DataFrame(property_type_ohe, columns=type_labels)

    # Merging the one-hot encoded columns back to the original dataframe
    df.reset_index(drop=True, inplace=True)
    df_ohe.reset_index(drop=True, inplace=True)
    df_merged = pd.concat([df, df_ohe], axis=1)

    return df_merged


# Calculate the correlation of the price with each of the k types of properties
def calculate_point_biserial_correlations(df, type_columns, price_column):
    correlations = {}
    for col in type_columns:
        # Calculate the point-biserial correlation coefficient and the p-value.
        corr, p_value = pointbiserialr(df[col], df[price_column])
        correlations[col] = corr  # Store the correlation coefficient.

    # Create a DataFrame from the correlations dictionary
    correlation_matrix = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation with Price'])

    return correlation_matrix[::-1]


# Extract the property type centers which were formed from k-Means
def extract_cluster_characteristics(df, cluster_col='property_type', feature_cols=None):
    if feature_cols is None:
        feature_cols = ['guests', 'beds', 'bedrooms', 'baths']

    cluster_info = []
    for cluster in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster]
        features_info = {'Type': f"Type {cluster + 1}"}
        for feature in feature_cols:
            most_common = cluster_data[feature].mode()[0]  # The most common value (mode) in each feature
            features_info[feature] = most_common
        cluster_info.append(features_info)

    df = pd.DataFrame(cluster_info)
    df.set_index('Type', inplace=True)
    return df


# Visualize the property type and price correlation
def visualize_type_vs_price_corr(collection):
    num_clusters = st.sidebar.number_input('Number of property types', value=5)

    # Fetch all documents from collection
    agg_pipeline = [
        {
            "$match": {
                "price": {"$ne": None},
                "details.guests": {"$ne": None},
                "details.beds": {"$ne": None},
                "details.bedrooms": {"$ne": None},
                "details.baths": {"$ne": None},
            }
        },
        {
            "$project": {
                "price": 1,
                "guests": "$details.guests",
                "beds": "$details.beds",
                "bedrooms": "$details.bedrooms",
                "baths": "$details.baths",
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    # Selecting the features for clustering
    features = df[['guests', 'beds', 'bedrooms', 'baths']]

    # Cluster the properties into k clusters
    df = cluster_properties(df, features, num_clusters)

    type_columns = [col for col in df.columns if 'Type' in col]
    # Calculate the type/price correlation matrix
    correlation_matrix = calculate_point_biserial_correlations(df, type_columns, 'price')
    # Extract the cluster characteristics
    cluster_characteristics = extract_cluster_characteristics(df)

    st.subheader('Point-Biserial Correlation Matrix')

    # Use Plotly to create an interactive heatmap
    fig = ff.create_annotated_heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns.tolist(),
        y=correlation_matrix.index.tolist(),
        annotation_text=correlation_matrix.round(2).values,
        showscale=True,
        colorscale='RdBu_r',
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=800, height=500, font_size=25)

    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=False)

    st.divider()
    st.subheader("Property Type Details")
    st.table(cluster_characteristics)


# ==================================================================================


# 2. Functions for finding top-10 and bottom-10 rated stays
# ==================================================================================
# Extract the top-k and bottom-k rated stays, with relation to the number of reviews
def filter_rated_stays(collection, num_ratings, min_num_reviews):
    # Fetch all documents from collection which have at least min_num_reviews reviews
    agg_pipeline = [
        {
            "$match": {
                "number_reviews": {"$gte": min_num_reviews},
                "review_index": {"$ne": None},
                "details.guests": {"$ne": None},
                "details.beds": {"$ne": None},
                "details.bedrooms": {"$ne": None},
                "details.baths": {"$ne": None},
            }
        },
        {
            "$project": {
                "review_index": 1,
                "number_reviews": 1,
                "guests": "$details.guests",
                "beds": "$details.beds",
                "bedrooms": "$details.bedrooms",
                "baths": "$details.baths",
                "host": 1
            }
        },
        {
            "$sort": {
                "review_index": -1
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    top_rated = df.head(num_ratings)
    bottom_rated = df.tail(num_ratings)

    return top_rated, bottom_rated


# Visualize the top-k and bottom-k rated stays
def visualize_rated_stays(collection):
    min_num_reviews = st.sidebar.number_input('Minimum number of reviews', value=5)
    top_rated, bottom_rated = filter_rated_stays(collection, 10, min_num_reviews)

    # Adjust indices to start from 1 for display purposes
    top_rated = top_rated.reset_index(drop=True)
    bottom_rated = bottom_rated.reset_index(drop=True)
    top_rated.index += 1
    bottom_rated.index += 1

    # Assign a group for plotting purposes
    top_rated['Group'] = 'Top'
    bottom_rated['Group'] = 'Bottom'

    # Combine the top and bottom rated for plotting
    combined = pd.concat([bottom_rated, top_rated]).reset_index(drop=True)

    # Create a unique position mapping for x-axis based on unique combinations
    combined['Position'] = combined.apply(lambda x: f"{x['review_index']}_{x['number_reviews']}_{x['host']}", axis=1)
    unique_positions = combined['Position'].unique()
    position_mapping = {pos: i for i, pos in enumerate(unique_positions)}

    combined['Position'] = combined['Position'].map(position_mapping)

    st.subheader("Top Rated Stays")
    st.table(top_rated[['guests', 'beds', 'bedrooms', 'baths', 'review_index', 'number_reviews', 'host']])
    st.divider()

    st.subheader("Bottom Rated Stays")
    st.table(bottom_rated[['guests', 'beds', 'bedrooms', 'baths', 'review_index', 'number_reviews', 'host']])
    st.divider()

    st.subheader("Top & Bottom Rated Hosts")
    # Add a comment above the bar plot
    st.markdown("""
       **Chart Explanation**: Each bar represents the number of reviews for a property. The properties on the left are the bottom-rated, ordered by their review index and number of reviews. The properties on the right are the top-rated, following the same ordering. The top-rated properties are shown in light blue and the bottom-rated properties in light grey.
       """)

    # Plotting the bar chart
    fig = px.histogram(combined, x="host", y="number_reviews", color="Group", height=600)

    st.plotly_chart(fig, use_container_width=True)


# ==================================================================================


# 3. Functions for most common characteristics and correlation with price
# ==================================================================================
# Find the most common k characteristics
def find_most_common_characteristics(collection, num_characteristics):
    # Fetch all documents from collection
    agg_pipeline = [
        {
            "$match": {
                "characteristics": {"$ne": []},
            }
        },
        {
            "$project": {
                "characteristics": 1
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    # Flatten the characteristics into 1d list
    characteristics = []
    for chars in df["characteristics"].to_numpy():
        for c in chars:
            characteristics.append(c)

    # Find the k most common characteristics
    most_common = []
    for char in Counter(characteristics).most_common(num_characteristics):
        most_common.append(char[0])

    return most_common


# Visualize the most common characteristics and their correlation with the price
def visualize_most_common_characteristics_price(collection):
    num_characteristics = st.sidebar.number_input('Number of most common characteristics', value=5)
    most_common_chars = find_most_common_characteristics(collection, num_characteristics)

    most_common = st.sidebar.selectbox(
        "Select characteristic",
        most_common_chars,
        0
    )

    # Fetch all documents from collection and check if they have characteristic or not
    agg_pipeline = [
        {
            "$match": {
                "price": {"$ne": None},
            }
        },
        {
            "$project": {
                "_id": 0,
                "Price": "$price",
                most_common: {
                    "$cond": [
                        {"$in": [most_common, "$characteristics"]},
                        "Yes",
                        "No"
                    ]
                }
            }
        }
    ]
    documents = collection.aggregate(agg_pipeline)

    df = pd.DataFrame(documents)

    fig = px.box(df, y="Price", x=most_common, color=most_common, points="all")

    fig.update_layout(
        autosize=False,
        height=700,
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# ==================================================================================


# 4. Functions for displaying general statistics per region
# ==================================================================================
# Visualize general statistics for the properties per region
def visualize_general_statistics_per_region(collection):
    options = st.sidebar.multiselect(
        'Select attributes to compare',
        ['Price', 'Review Index', 'Number of Reviews', 'Superhost', 'Guest Favorite'],
        ['Price', 'Review Index'])

    if len(options) == 0:
        st.warning("Warning: No attributes selected", icon="⚠️")
        return

    # Prepare a group query based on the selected attributes
    group_by = {"_id": "$region"}
    if "Price" in options:
        group_by['Average Price'] = {"$avg": "$price"}
    if "Review Index" in options:
        group_by['Average Review Index'] = {"$avg": "$review_index"}
    if "Number of Reviews" in options:
        group_by['Average Number of Reviews'] = {"$avg": "$number_reviews"}
    if "Superhost" in options:
        group_by['Number of Superhosts'] = {"$sum": {"$cond": ["$superhost", 1, 0]}}
    if "Guest Favorite" in options:
        group_by['Number Guests Favorite'] = {"$sum": {"$cond": ["$guest_favorite", 1, 0]}}

    # Fetch all documents from collection and calculate the average/sum for the selected attributes
    agg_pipeline = [
        {
            "$match": {
                "region": {"$ne": None},
                "price": {"$ne": None},
                "review_index": {"$ne": None},
                "number_reviews": {"$ne": None},
                "superhost": {"$ne": None},
                "guest_favorite": {"$ne": None},
            }
        },
        {
            "$group": group_by
        }
    ]
    documents = collection.aggregate(agg_pipeline)

    df = pd.DataFrame(documents)

    keys = df.keys().tolist()
    keys.remove('_id')

    # Normalize the attribute values to [0, 1] by dividing by the max value of each column
    df_normalized = df.copy()
    for key in keys:
        df_normalized[key] /= df_normalized[key].max()

    # Rearrange the data in order to be displayed in a bar chart
    bar_df = pd.DataFrame()
    for key in keys:
        data = df_normalized[[key, '_id']]
        for index, row in data.iterrows():
            df2 = {'Attribute': key, 'Value': row[key], 'Region': row['_id']}
            bar_df = bar_df.append(df2, ignore_index=True)

    # Create the bar chart
    fig = px.histogram(bar_df, x="Attribute", y="Value",
                       color='Region', barmode='group',
                       height=500)

    st.subheader("Normalized Attribute Statistics per Region")

    # Display the bar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Attribute Statistics per Region")

    df = df.rename(columns={"_id": "Region"})
    st.table(df.head())


# ==================================================================================


# 5. Functions for displaying a map of properties
# ==================================================================================
# Return a color based on the superhost condition
def get_color(superhost):
    # Define red for non-superhosts and green for superhosts
    return [255, 182, 193, 160] if superhost else [144, 238, 144, 160]  # Light pink and light green


# Find the central location from all properties
def find_central_location(collection):
    # Fetch all documents from collection
    agg_pipeline = [
        {
            "$match": {
                "location.coordinates": {"$ne": []},
            }
        },
        {
            "$project": {
                "lat": {"$arrayElemAt": ["$location.coordinates", 0]},
                "lng": {"$arrayElemAt": ["$location.coordinates", 1]},
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    # Return the mean latitude and longitude of the dataset
    return {"lat": df["lat"].mean(), "lng": df["lng"].mean()}


# Calculate general statistics for the properties within a distance from a central location
def calculate_statistics(collection, center_lat, center_lng, radius):
    # Fetch all documents from collection which are within a radius to the central property
    agg_pipeline = [
        {
            "$geoNear": {
                "near": {"type": "Point", "coordinates": [center_lat, center_lng]},
                "spherical": "true",
                "distanceField": "dist",
                "maxDistance": radius * 1000,  # in meters
            }
        },
        {
            "$match": {
                "price": {"$ne": None},
                "review_index": {"$ne": None},
                "superhost": {"$ne": None},
                "guest_favorite": {"$ne": None},
                "details.guests": {"$ne": None},
                "location.coordinates": {"$ne": []},
            }
        },
        {
            "$project": {
                "price": 1,
                "review_index": 1,
                "superhost": 1,
                "guest_favorite": 1,
                "guests": "$details.guests",
                "lat": {"$arrayElemAt": ["$location.coordinates", 0]},
                "lng": {"$arrayElemAt": ["$location.coordinates", 1]},
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    filtered_properties = pd.DataFrame(documents)

    # Calculate statistics
    if 'review_index' in filtered_properties.keys():
        avg_review_index = "{:.2f}".format(filtered_properties['review_index'].mean())
    else:
        avg_review_index = "n/a"

    if 'price' in filtered_properties.keys():
        avg_price = "{:.2f}".format(filtered_properties['price'].mean())
    else:
        avg_price = "n/a"

    if 'guests' in filtered_properties.keys():
        avg_guests = "{:.2f}".format(filtered_properties['guests'].mean())
    else:
        avg_guests = "n/a"

    if 'superhost' in filtered_properties.keys():
        percent_superhosts = "{:.2f}".format(
            (filtered_properties['superhost'].sum() / len(filtered_properties) * 100)) if len(
            filtered_properties) > 0 else 0
    else:
        percent_superhosts = "n/a"

    if 'guest_favorite' in filtered_properties.keys():
        percent_guest_favorites = "{:.2f}".format(
            (filtered_properties['guest_favorite'].sum() / len(filtered_properties) * 100)) if len(
            filtered_properties) > 0 else 0
    else:
        percent_guest_favorites = "n/a"

    st.divider()
    st.subheader("Selected Area General Statistics")

    # Display statistics
    annotated_text(
        "► Average Review Index: ",
        (f"{avg_review_index}", "Stars")
    )
    annotated_text(
        "► Average Price: ",
        (f"{avg_price}", "€")
    )
    annotated_text(
        "► Average Number of Guests: ",
        (f"{avg_guests}", "Guests")
    )
    annotated_text(
        "► Percentage of Superhosts: ",
        (f"{percent_superhosts}", "%")
    )
    annotated_text(
        "► Percentage of Guest's Favorites: ",
        (f"{percent_guest_favorites}", "%")
    )


# Visualize a map of the properties
def create_map(collection, initial_radius=1.0):
    # Fetch all documents from collection
    agg_pipeline = [
        {
            "$match": {
                "price": {"$ne": None},
                "review_index": {"$ne": None},
                "superhost": {"$ne": None},
                "guest_favorite": {"$ne": None},
                "details.guests": {"$ne": None},
                "location.coordinates": {"$ne": []},
            }
        },
        {
            "$project": {
                "price": 1,
                "review_index": 1,
                "superhost": 1,
                "guest_favorite": 1,
                "guests": "$details.guests",
                "lat": {"$arrayElemAt": ["$location.coordinates", 0]},
                "lng": {"$arrayElemAt": ["$location.coordinates", 1]},
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    central_property = find_central_location(collection)

    # Use Streamlit sliders to dynamically adjust the position of the circle and the marker
    lat = st.sidebar.slider("Select the Latitude of the circle's center", float(df['lat'].min()),
                            float(df['lat'].max()),
                            float(central_property['lat']))
    lng = st.sidebar.slider("Select the Longitude of the circle's center", float(df['lng'].min()),
                            float(df['lng'].max()),
                            float(central_property['lng']))
    radius = st.sidebar.slider("Radius (kilometers)", 0.1, 5.0, initial_radius, step=0.1)  # Slider in kilometers

    # Data for the properties layer
    data = [{
        'lat': row['lat'],
        'lng': row['lng'],
        'color': get_color(row['superhost'])
    } for index, row in df.iterrows()]

    # Layer for all properties
    layer = pdk.Layer(
        "ScatterplotLayer",
        data,
        get_position='[lng, lat]',
        get_color='color',
        get_radius=100,
    )

    # Circle layer positioned based on slider input
    circle_layer = pdk.Layer(
        "ScatterplotLayer",
        [{'lat': lat, 'lng': lng}],
        get_position='[lng, lat]',
        get_radius=radius * 1000,
        get_color='[135, 206, 250, 160]',  # Light blue color
    )

    # Marker that moves with the circle
    center_marker = pdk.Layer(
        "ScatterplotLayer",
        [{'lat': lat, 'lng': lng}],
        get_position='[lng, lat]',
        get_color='[135, 206, 250, 160]',  # Light blue color
        get_radius=120,  # Slightly larger to stand out
    )

    # View state automatically adjusts to center on the moved circle
    view_state = pdk.ViewState(latitude=lat, longitude=lng, zoom=11)

    # Render the map with updated layers
    map_ = pdk.Deck(layers=[layer, circle_layer, center_marker], initial_view_state=view_state)
    st.pydeck_chart(map_)

    # Calculate and display statistics for the properties within the circle
    calculate_statistics(collection, lat, lng, radius)


# ==================================================================================


# Functions for the ML Part of the project
# ==================================================================================
# Train the ML model for predicting the price
def train_model(df):
    X = df.drop(['price'], axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = model.predict(X_test)

    # Calculating the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Calculating the Mean Squared Error and R^2 score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing the metrics to the console
    print(f'Accuracy: {accuracy}\nMean Squared Error: {mse}\nR^2 Score: {r2}')

    return model


# Predict the price of a property based on the selected attribute values
def price_estimation(collection):
    # Fetch all documents from collection
    agg_pipeline = [
        {
            "$match": {
                "price": {"$ne": None},
                "superhost": {"$ne": None},
                "guest_favorite": {"$ne": None},
                "review_index": {"$ne": None},
                "number_reviews": {"$ne": None},
                "details.guests": {"$ne": None},
                "details.beds": {"$ne": None},
                "details.bedrooms": {"$ne": None},
                "details.baths": {"$ne": None},
                "location.coordinates": {"$ne": []},
            }
        },
        {
            "$project": {
                "_id": 0,
                "price": 1,
                "superhost": 1,
                "guest_favorite": 1,
                "review_index": 1,
                "number_reviews": 1,
                "guests": "$details.guests",
                "beds": "$details.beds",
                "bedrooms": "$details.bedrooms",
                "baths": "$details.baths",
                "lat": {"$arrayElemAt": ["$location.coordinates", 0]},
                "lng": {"$arrayElemAt": ["$location.coordinates", 1]},
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)

    # Creation of Dataframe
    df = pd.DataFrame(documents)

    # Input fields for user to enter property features
    guests = st.number_input('Number of Guests', min_value=1, value=1)
    beds = st.number_input('Number of Beds', min_value=1, value=1)
    bedrooms = st.number_input('Number of Bedrooms', min_value=1, value=1)
    baths = st.number_input('Number of Baths', min_value=1, value=1)
    superhost = st.checkbox('Superhost')
    guest_favorite = st.checkbox('Guest Favourite')
    review_index = st.slider('Review Index', 0.0, 5.0, 4.5)
    number_reviews = st.number_input('Number of Reviews', min_value=0, value=0)

    # Prepare the feature for prediction based on user input
    user_features = pd.DataFrame(
        [[superhost, guest_favorite, review_index, number_reviews, guests, beds, bedrooms, baths, ]],
        columns=['superhost', 'guest_favorite', 'review_index', 'number_reviews', 'guests', 'beds', 'bedrooms',
                 'baths'])

    # Assuming lat-lon to be average values from the data for the demonstration purpose
    user_features['lat'] = df['lat'].mean()
    user_features['lng'] = df['lng'].mean()

    if st.button('Predict Price'):
        model = train_model(df)
        prediction = model.predict(user_features)
        st.success(f'The estimated price for the property is: €{prediction[0]}')


# ==================================================================================


def main():
    pd.set_option("display.max.columns", None)

    st.set_page_config(layout="wide")

    # Load the MongoDB collection
    start_time = time.time()
    collection = load_mongodb_collection()
    print("--- Load MongoDB: %s seconds ---" % (time.time() - start_time))

    # Streamlit
    st.title('Airbnb Scraping & Analysis')
    st.sidebar.title('Mini Project I: Web Scraping & Analysis')
    st.sidebar.markdown('_By **Filitsa Ioanna Kouskouveli** and ***Vasilis Andritsoudis***_')
    st.sidebar.divider()

    # Create tabs
    tab_1, tab_2, tab_3, tab_4, tab_5, tab_ml = st.tabs(
        ["Price - Type Correlation", "Rated Stays", "Common Characteristics", "General Statistics", "Map of Properties",
         "Price Estimation"])

    with tab_1:
        with st.spinner("Loading..."):
            st.header('Price and Property Type Correlation')
            st.divider()
            st.sidebar.subheader('Price and Property Type Correlation')
            start_time = time.time()
            visualize_type_vs_price_corr(collection)
            print("--- Visualize property type / price correlation matrix: %s seconds ---" % (time.time() - start_time))
            st.sidebar.divider()

    with tab_2:
        with st.spinner("Loading..."):
            st.header('Rated Stays')
            st.divider()
            st.sidebar.subheader('Rated Stays')
            start_time = time.time()
            visualize_rated_stays(collection)
            print("--- Visualize rated stays: %s seconds ---" % (time.time() - start_time))
            st.sidebar.divider()

    with tab_3:
        with st.spinner("Loading..."):
            st.header('Price Correlation with Common Characteristics')
            st.divider()
            st.sidebar.subheader('Price Correlation with Common Characteristics')
            start_time = time.time()
            visualize_most_common_characteristics_price(collection)
            print("--- Visualize most common characteristics and price correlation: %s seconds ---" % (
                        time.time() - start_time))
            st.sidebar.divider()

    with tab_4:
        with st.spinner("Loading..."):
            st.header('General Statistics per Region')
            st.divider()
            st.sidebar.header('General Statistics per Region')
            start_time = time.time()
            visualize_general_statistics_per_region(collection)
            print("--- Visualize general statistics per region: %s seconds ---" % (time.time() - start_time))
            st.sidebar.divider()

    with tab_5:
        with st.spinner("Loading..."):
            st.header('Map of Properties')
            st.divider()
            st.sidebar.header('Map of Properties')
            start_time = time.time()
            create_map(collection)
            print("--- Create map of properties: %s seconds ---" % (time.time() - start_time))

    with tab_ml:
        with st.spinner("Loading..."):
            st.header('Price Estimation')
            st.divider()
            start_time = time.time()
            price_estimation(collection)
            print("--- Price estimation: %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()
