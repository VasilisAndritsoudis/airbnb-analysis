## Airbnb Scraping & Analysis
Andritsoudis Vasilis & Filitsa Ioanna Kouskouveli

### Introduction: Why Are We Here?

Welcome to our friendly journey into the Airbnb’s listings, where we'll be extracting valuable data insights. Why? Well, besides the chance to sharpen our Python skills, we will also be given the chance to uncover any useful findings about regions of Thessaloniki, a popular destination in Greece.

### What We’re Gathering

* Price Per Night: discovering the real cost behind those listings
* Properties' Spaces: details like beds, bedrooms and bathroom logistics
* Host Insights: evaluating Superhosts and Guests’ favorites
* Ratings and Reviews: review index and number of reviews 
* Special Features: like the hosts’ names, properties’ characteristics, if available
* Coordinates: making sure we know exactly where these places are

### Strategy Overview

Our game plan includes:

* Data Collection: leveraging Selenium for smart automation and HTML extraction
* Data Exploitation: using Python for data processing, visually appealing and comprehensive data presentation and data modeling.

### Setting Up the Necessary Tools

#### Setting Up Selenium

Why Use It?: Automating web interactions not only saves time but also brings a bit of magic to our data collection efforts.

Installation:

```sh
pip install selenium
```

Remember, downloading the correct WebDriver is like choosing the right tool for the right job-essential for smooth operations.

#### PyMongo: Storing Our Findings
Why It’s Useful: PyMongo acts as a safe where we can store all the valuable insights we collect, organized and secure in MongoDB.

```sh
pip install pymongo
```

#### Streamlit: Bringing Data to Life

Why We Adore It: Streamlit allows us to create interactive and beautiful visualizations, making our data not only accessible but also engaging.

```sh
pip install streamlit
```

A custom text highlighting package we will use for making the Streamlit UI more interesting.

```sh
pip install st-annotated-text
```

### What's Next?

Given the right tools and an analytical plan, we’re ready to tackle our data analysis project. This guide will help you navigate through Airbnb’s data landscape, organize your findings in MongoDB, present them in an appealing way with Streamlit and further model it. 


Time to dive in and explore!

## Airbnb Data Collection: A Tech-Savvy Approach

### Web Scraping Process Explained

The *scraping.ipynb* is not just a bunch of Python lines; it's a carefully designed script to navigate Airbnb’s platform and carefully extract data. Let's unpack the process:

**WebDriver Initialization**: The script starts by breathing life into the Selenium WebDriver, providing a means of simulating real-world interaction with a web browser.

```python
browser = webdriver.Chrome(options=options)
browser.get('https://www.airbnb.com')
```

**Automated Navigation**: It then proceeds to the heart of Airbnb, entering search terms and engaging with UI elements to find the listings we're interested in. For the region Neapoli-Sikies, in Thessaloniki, we would have for example:

```python
elem.send_keys('Neapoli-Sikies, Greece' + Keys.RETURN)
```

**Data Extraction Functions**: Each function acts as a specialized tool, engineered to extract distinct slices of information from a listing's detail-rich page.

```python
def extract_price(browser):
  # Extract the price per night
  try:
    # If no discount
    return float(browser.find_element(By.XPATH, "//*[@class='_tyxjp1']").get_attribute('innerHTML').split(';')[1].split('&')[0])
  except:
    try:
      # If discount
      return float(browser.find_element(By.XPATH, "//*[@class='_1y74zjx']").get_attribute('innerHTML').split(';')[1].split('&')[0])
    except:
      return None
```

**Data Assembly Line**: Extracted data is methodically assembled into a structured dictionary format, then batch-inserted into MongoDB, ensuring our data is systematically cataloged.

```python
# Extract the price
price = extract_price(browser)
db_object['price'] = price

...

# Append to the list of properties
db_objects.append(db_object)

...

# Insert all objects to MongoDB
collection.insert_many(db_objects)
```

### Textual Data Preprocessing

As for the data to take their textual/clean form, the notebook serves as an editor, scrutinizing raw text to refine and standardize it for analytical consumption:

**Textual Cleanup**: Our code scrubs through text, eliminating extraneous characters and standardizing entries to ensure consistency across the dataset.

```python
# Search for elements that contain "Hosted by"
elem = browser.find_element(By.XPATH, "//*[contains(text(), 'Hosted by')]")
# Use regex to split the text by "Hosted by " and capture the following text
host_name_match = re.search(r'(?:Hosted by)(.+)', elem.text)
```

**Transformation Into Analytics-Ready Data**: Converting text to numerical data where appropriate, the script parses and extracts values such as review counts and indexes for straightforward analysis.

```python
review_index = float(numbers[0].replace(',', '.'))
```

### Final Data Structure and MongoDB Insertion

In the culmination of the web scraping process, our script crafts a well-organized DataFrame. Each row of this DataFrame corresponds to an individual property with the following columns: **price** (the nightly rate to stay at the property), **superhost** (a boolean indicating whether the host is recognized as a Superhost), **guest_favorite** (a boolean reflecting whether the property is a favorite among guests), **review_index** (a numerical value representing the average review score), **number_reviews** (the count of how many reviews the property has received), **host** (the name of the individual or entity hosting the property) **characteristics** (a list of notable features or amenities associated with the property), **guests** (the maximum number of guests allowed at the property), **beds** (the number of beds available bedrooms) **bedrooms** (the number of bedrooms included), **baths** (the number of bathrooms provided), **latitude** and **longitude** (geographical coordinates marking the property's location).

After preprocessing and structuring this data, the script pushes the entire collection of listings as documents into a MongoDB collection. Here's a brief overview of how that's executed:

```python
# Extract the property information
db_object = create_db_object(browser, region)

# Append to the list of properties
db_objects.append(db_object)

... 

collection.insert_many(db_objects)
```

The *insert_many* method is invoked on the collection object from PyMongo, which sends each listing's details to our MongoDB database, where it is stored securely and ready for further analysis or querying. This method is called within the loop that processes each listing, ensuring real-time storage of data as it's scraped.

Below is an example of a JSON record that is saved to the database:

```json
{
    "_id": "661bd6b70e39a4efc9d5cf56",
    "price": 39,
    "region": "Kalamaria",
    "host": "Dimitris",
    "superhost": false,
    "review_index": 4.67, 
    "number_reviews": 100,
    "guest_favorite": false,
    "characteristics": ["Self check-in"],
    "details": {
        "guests": 2,
        "beds": 1,
        "bedrooms": 1,
        "baths": 1
    },
    "location": {
        "type": "Point",
        "coordinates": [40.58439817044409, 22.964]
    }
}
```

## Data Analysis for Airbnb Data

### Backbone of the Code

The provided Python code constitutes a multi-functional script designed to interact with a MongoDB collection, containing Airbnb property data. It facilitates the analysis and visualization of this data using a combination of Python’s analytical libraries and the Streamlit framework. The key activities involve:

1. **Data Clustering and Correlation Analysis**: Identifying property types through clustering and assessing their relationship with pricing.
2. **Rating Analysis**: Filtering properties based on their reviews to identify top-rated and bottom-rated stays.
3. **Most Common Characteristics and Correlation with Price**: Analyzing how frequently certain property features appear and their impact on property pricing.
4. **Displaying General Statistics per Region**: Aggregating data by region to show general trends and metrics such as average price and review scores.
5. **Mapping Visualization**: Plotting property locations on a map with Streamlit to provide insights into the dataset in reference to geo-location.

### Correlation Matrix - Property Type and Price

The main utilized function is *visualize_type_vs_price_corr*. It undertakes a series of operations aimed at understanding the correlation between the **types of properties** and their listed **prices**.

**Types of properties** are defined through k-means clustering, a machine learning algorithm. This algorithm groups properties into different 'types' based on similarities in their features, specifically the number of guests they can accommodate, the number of beds, bedrooms, and baths available. The *num_clusters* parameter is selectable from the UI and determines how many distinct types or groups the algorithm will create.

```python
num_clusters = st.sidebar.number_input('Number of property types', value=5)

...

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['property_type'] = kmeans.fit_predict(features)
```

Before clustering documents are retrieved from the MongoDB collection. K-means is then run, where each cluster center depicts a type of property, consisting of a specific number of guests, beds, bedrooms and baths. The types are then used to examine how each group correlates with the property prices, giving insight into the influence of property features on pricing. Afterwards, the script computes the point-biserial correlation coefficients to measure the strength and direction of the association between property types (categorical) and their prices (continuous), Finally it employs an interactive heatmap to graphically represent these correlations. The heatmap shows clearly the correlation of each property type with the property price.

```python
# Fetch all documents from collection
agg_pipeline = [
  {"$match": {"price": {"$ne": None},"details.guests": {"$ne": None},"details.beds": {"$ne": None},"details.bedrooms": {"$ne": None},"details.baths": {"$ne": None}}},
  {"$project": {"price": 1,"guests": "$details.guests","beds": "$details.beds","bedrooms": "$details.bedrooms","baths": "$details.baths"}}
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

# Use Plotly to create an interactive heatmap
fig = ff.create_annotated_heatmap(z=correlation_matrix.values, x=correlation_matrix.columns.tolist(), y=correlation_matrix.index.tolist(),annotation_text=correlation_matrix.round(2).values, showscale=True, colorscale='RdBu_r')

fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), width=800, height=500, font_size=25)

# Display the heatmap in Streamlit
st.plotly_chart(fig, use_container_width=False)
```

### Top-10 and Bottom-10 Rated Stays

The *visualize_rated_stays* function is designed to identify and display the properties with the highest and lowest guest satisfaction scores. It does so by filtering out properties based on a user-specified minimum number of reviews to ensure that the ratings are reliable.

Users can input the minimum number of reviews through a Streamlit sidebar element. This input acts as a threshold, ensuring that only those properties with a review count above this number are considered in the analysis. This parameterizable input allows users to control the strictness of the filtering criteria.

Only documents that satisfy the filtering criteria are fetched from the MongoDB collection and are then sorted based on their review index (a numerical representation of their average rating). Once the data is filtered, the function slices out the top 10 and bottom 10 for display.

```python
# Extract the top-k and bottom-k rated stays, with relation to the number of reviews
def filter_rated_stays(collection, num_ratings, min_num_reviews):
  # Fetch all documents from collection which have at least min_num_reviews reviews
  agg_pipeline = [
    {"$match": {"number_reviews": {"$gte": min_num_reviews}, "review_index": {"$ne": None}, "details.guests": {"$ne": None}, "details.beds": {"$ne": None}, "details.bedrooms": {"$ne": None}, "details.baths": {"$ne": None}}},
    {"$project": {"review_index": 1, "number_reviews": 1, "guests": "$details.guests", "beds": "$details.beds", "bedrooms": "$details.bedrooms", "baths": "$details.baths", "host": 1}},
    { "$sort": { "review_index": -1}}
  ]

  documents = collection.aggregate(agg_pipeline)

  # Creation of Dataframe
  df = pd.DataFrame(documents)

  top_rated = df.head(num_ratings)
  bottom_rated = df.tail(num_ratings)

  return top_rated, bottom_rated

...

min_num_reviews = st.sidebar.number_input('Minimum number of reviews', value=5)
top_rated, bottom_rated = filter_rated_stays(collection, 10, min_num_reviews)

...

st.subheader("Top Rated Stays")
st.table(top_rated[['guests', 'beds', 'bedrooms', 'baths', 'review_index', 'number_reviews', 'host']])
```

In this snippet, *min_num_reviews* is set by the user via the Streamlit interface. The *filter_rated_stays* function then uses this value to retrieve, sort and select the top and bottom properties based on their ratings. The resulting lists of properties are then presented in the Streamlit app using tables that include essential details such as guest capacity, number of beds, bedrooms, baths, review index, number of review and the host's name, allowing users to easily compare the highest and lowest-rated stays.

### Most Common Characteristics and Correlation with Price

The function *visualize_most_common_characteristics_price* aims to analyze and visualize the impact of the most common characteristics (features like amenities or services offered by the property) on the pricing of listings. This is conducted through a filtering process that identifies the most common characteristics across all entries and then assesses their presence's correlation with property prices using a box plot for a visual summary. The number of most common characteristics is a variable selectable from the UI.

```python
# Find the most common k characteristics
def find_most_common_characteristics(collection, num_characteristics):
    # Fetch all documents from collection
    agg_pipeline = [
        {"$match": {"characteristics": {"$ne": []}}},
        {"$project": {"characteristics": 1}}
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
```

In fact, the *find_most_common_characteristics* function retrieves characteristics from the MongoDB collection, flattens the list of characteristics from all documents and identifies the most common ones. 

```python
# Visualize the most common characteristics and their correlation with the price
def visualize_most_common_characteristics_price(collection):
    num_characteristics = st.sidebar.number_input('Number of most common characteristics', value=5)
    most_common_chars = find_most_common_characteristics(collection, num_characteristics)
    most_common = st.sidebar.selectbox("Select characteristic", most_common_chars, 0)

    # Fetch all documents from collection and check if they have characteristic or not
    agg_pipeline = [
        {"$match": {"price": {"$ne": None}}},
        {"$project": {
                "_id": 0, "Price": "$price",
                most_common: {"$cond": [{"$in": [most_common, "$characteristics"]},"Yes", "No"]}
            }
        }
    ]

    documents = collection.aggregate(agg_pipeline)
    df = pd.DataFrame(documents)
    fig = px.box(df, y="Price", x=most_common, color=most_common, points="all")
    fig.update_layout(autosize=False, height=700)

    # Display the heatmap in Streamlit
    st.plotly_chart(fig, use_container_width=True)
```

Thus, the *visualize_most_common_characteristics_price* function allows users to select one of these characteristics (each time) and show the distribution of prices for listings with and without the selected characteristic. It does that by using Plotly to create a box plot. This visual analysis helps identify if the presence of a characteristic has a noticeable effect on pricing.

### Displaying General Statistics per Region

The function *visualize_general_statistics_per_region* computes and visualizes key statistics aggregated by region. This includes averages and sums of selected metrics such as price and review indexes. This visualization aims to provide insights into regional differences and trends within the dataset.

```python
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
    if "Superhost" in options:
        group_by['Number of Superhosts'] = {"$sum": {"$cond": ["$superhost", 1, 0]}}

    ...

    # Fetch all documents from collection and calculate the average/sum for the selected attributes
    agg_pipeline = [
      {"$match": {"region": {"$ne": None},"price": {"$ne": None},"superhost":{"$ne": None}}},
      {"$group": group_by}
    ]

    documents = collection.aggregate(agg_pipeline)
    df = pd.DataFrame(documents)

    ...

    # Create the bar chart
    fig = px.histogram(bar_df, x="Attribute", y="Value", color='Region', barmode='group', height=500)
    st.subheader("Normalized Attribute Statistics per Region")

    # Display the bar chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)
```

This function allows users to select which attributes to compare across regions using a sidebar multi-select box. It constructs a MongoDB aggregation pipeline to calculate average or sum statistics grouped by the region. The results are then visualized using a Plotly bar chart, which provides a comparative view of the metrics across different regions, highlighting regional market trends and anomalies in the dataset.

### Map of Properties

The *create_map* function incorporates user input to adjust the visualization dynamically. It leverages Streamlit's sidebar sliders that allow users to select a geographical point by adjusting the latitude and longitude values. Additionally, users can specify the radius of interest around this point to focus on properties within that area.

As the user interacts with the sliders, the map view updates in real-time:

* the **latitude** and **longitude sliders** change the central point of the map, essentially moving the visual focus to different areas
* the **radius slider** adjusts the size of a circle on the map that highlights properties within that specified radius from the central point, allowing users to examine a localized cluster of properties

The map then displays properties falling within the circle, marked with color-coded points to distinguish between different types, such as superhosts and regular hosts. The information alongside the map updates to reflect statistics about the properties in the selected area, including average price, average review index and the percentage of superhosts and guest favorites.

```python
# Use Streamlit sliders to dynamically adjust the position of the circle and the marker
lat = st.sidebar.slider("Select the Latitude of the circle's center", float(df['lat'].min()), float(df['lat'].max()), float(central_property['lat']))
lng = st.sidebar.slider("Select the Longitude of the circle's center", float(df['lng'].min()), float(df['lng'].max()), float(central_property['lng']))
radius = st.sidebar.slider("Radius (kilometers)", 0.1, 5.0, initial_radius, step=0.1)  # Slider in kilometers

# Circle layer positioned based on slider input
circle_layer = pdk.Layer(
    "ScatterplotLayer",
    [{'lat': lat, 'lng': lng}],
    get_position='[lng, lat]',
    get_radius=radius * 1000,
    get_color='[135, 206, 250, 160]',  # Light blue color
)
```
In this snippet, the *circle_layer* is updated based on the latitude, longitude, and radius parameters. The circle's center moves according to the lat and lng values, and its size changes with the radius, directly responding to the user's slider adjustments.

General statistics are then calculated for the properties located inside the circle, by first retrieving them from the MongoDB collection using Geo queries. After the properties are retrieved, averages are calculated for the price, review index, guests, superhost and guest favorite attributes.

```python
# Calculate general statistics for the properties within a distance from a central location
def calculate_statistics(collection, center_lat, center_lng, radius):
  # Fetch all documents from collection which are within a radius to the central property
  agg_pipeline = [
    {"$geoNear": {"near": {"type": "Point", "coordinates": [center_lat, center_lng]}, "spherical": "true", "distanceField": "dist", "maxDistance": radius * 1000}},
    {"$match": {"price": {"$ne": None}, "review_index": {"$ne": None}, "location.coordinates": {"$ne": []}}},
    {"$project": {"price": 1, "review_index": 1, "lat": {"$arrayElemAt": ["$location.coordinates", 0]}, "lng": {"$arrayElemAt": ["$location.coordinates", 1]}}}
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
  
  ... 
```

## Data Modeling

The ML component of the project is centered around using a Random Forest classifier to predict Airbnb rental prices based on property features. This part of the code is integrated within a Streamlit app, which interacts with users allowing them to input features and receive estimated prices.

### An overview of the ML Workflow

The features, that were decided to be used, included guests, beds, bedrooms, baths, superhost status, guest favorite status, review index, number of reviews and geographical coordinates (latitude and longitude). Following the feature selection, the data is split into training and test datasets and a Random Forest classifier is trained on the training data. The model makes predictions on the test dataset and performance metrics like accuracy, MSE (Mean Squared Error), and R² score are calculated to evaluate the model.

The user interaction takes place through the Streamlit app, as users can input the features of a property. After the input of features’ values by the user, the app uses the trained model to estimate and display the rental price based on the input.

```python
def price_estimation(collection):
  guests = st.number_input('Number of Guests', min_value=1, value=1)
  beds = st.number_input('Number of Beds', min_value=1, value=1)

  ...

  # Prepare the feature for prediction based on user input
  user_features = pd.DataFrame([[guests, beds]], columns=['guests', 'beds'])

  if st.button('Predict Price'):
    model = train_model(df)
    prediction = model.predict(user_features)
    st.success(f'The estimated price for the property is: €{prediction[0]}')
```

User Inputs: Streamlit widgets collect property features from the user. These features match those used during model training.

Price Prediction: Once the user clicks the "Predict Price" button, the model predicts the price based on the input features, and the estimated price is displayed.

## Putting it all together

This snippet shows the high-level workflow from data loading, preprocessing, model training, to setting up the Streamlit interface for user interaction. It highlights the sequence of function calls for processing and using the data, encapsulating the project's end-to-end workflow within the *main* function.

```python
def main():
  pd.set_option("display.max.columns", None)

  collection = load_mongodb_collection()

  # Streamlit
  st.set_page_config(layout="wide")
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
      visualize_type_vs_price_corr(collection)
      st.sidebar.divider()
  
  ...
```

## In Summary

Diving into the world of Airbnb listings in Thessaloniki, this Medium article zips you through scraping, cleaning, analyzing and even predicting rental prices with Python and tools like Selenium and MongoDB. We started off by setting up Selenium to automate the dull stuff - navigating and extracting data from Airbnb’s website. With the data in hand, we cleaned it up and got it ready for analysis and modeling. Using Streamlit, we presented different plots and visuals, trying to shed light on what’s really going on with Airbnb prices and trends in the city. Then we employed machine learning to predict rental prices, the user inputs a property’s features into our model and see magic happen - rental price predictions pop out. And guess what? You can find all the nitty-gritty details, code, in our GitHub repo. Go on, give it a click and start exploring the web data world!