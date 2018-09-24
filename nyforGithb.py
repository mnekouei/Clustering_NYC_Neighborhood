
# coding: utf-8

# <h1 align=center><font size = 5>Segmenting and Clustering Neighborhoods in New York City</font></h1>

# ## Introduction
# 
# In this lab, you will learn how to convert addresses into their equivalent latitude and longitude values. Also, you will use the Foursquare API to explore neighborhoods in New York City. You will use the **explore** function to get the most common venue categories in each neighborhood, and then use this feature to group the neighborhoods into clusters. You will use the *k*-means clustering algorithm to complete this task. Finally, you will use the Folium library to visualize the neighborhoods in New York City and their emerging clusters.

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>
# 
# 1. <a href="#item1">Download and Explore Dataset</a>
# 
# 2. <a href="#item2">Explore Neighborhoods in New York City</a>
# 
# 3. <a href="#item3">Analyze Each Neighborhood</a>
# 
# 4. <a href="#item4">Cluster Neighborhoods</a>
# 
# 5. <a href="#item5">Examine Clusters</a>    
# </font>
# </div>

# Before we get the data and start exploring it, let's download all the dependencies that we will need.

# In[3]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system("conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported.')


# <a id='item1'></a>

# ## 1. Download and Explore Dataset

# Neighborhood has a total of 5 boroughs and 306 neighborhoods. In order to segement the neighborhoods and explore them, we will essentially need a dataset that contains the 5 boroughs and the neighborhoods that exist in each borough as well as the the latitude and logitude coordinates of each neighborhood. 
# 
# Luckily, this dataset exists for free on the web. Feel free to try to find this dataset on your own, but here is the link to the dataset: https://geo.nyu.edu/catalog/nyu_2451_34572

# For your convenience, I downloaded the files and placed it on the server, so we can simply run a `wget` command and access the data. So let's go ahead and do that.

# In[4]:


get_ipython().system("wget -q -O 'newyork_data.json' https://ibm.box.com/shared/static/fbpwbovar7lf8p5sgddm06cgipa2rxpe.json")
print('Data downloaded!')


# #### Load and explore the data

# Next, let's load the data.

# In[5]:


with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)


# Notice how all the relevant data is in the *features* key, which is basically a list of the neighborhoods. So, let's define a new variable that includes this data.

# In[6]:


neighborhoods_data = newyork_data['features']


# Let's take a look at the first item in this list.

# In[7]:


neighborhoods_data[0]


# #### Tranform the data into a *pandas* dataframe

# The next task is essentially transforming this data of nested Python dictionaries into a *pandas* dataframe. So let's start by creating an empty dataframe.

# In[8]:


# define the dataframe columns
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiate the dataframe
neighborhoods = pd.DataFrame(columns=column_names)


# Take a look at the empty dataframe to confirm that the columns are as intended.

# In[9]:


neighborhoods


# Then let's loop through the data and fill the dataframe one row at a time.

# In[10]:


for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    neighborhoods = neighborhoods.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)


# Quickly examine the resulting dataframe.

# In[11]:


neighborhoods.head()


# And make sure that the dataset has all 5 boroughs and 306 neighborhoods.

# In[12]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)


# #### Use geopy library to get the latitude and longitude values of New York City.

# In[13]:


address = 'New York City, NY'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# #### Create a map of New York with neighborhoods superimposed on top.

# In[14]:


# create map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_newyork)  
    
map_newyork


# **Folium** is a great visualization library. Feel free to zoom into the above map, and click on each circle mark to reveal the name of the neighborhood and its respective borough.

# However, for illustration purposes, let's simplify the above map and segment and cluster only the neighborhoods in Manhattan. So let's slice the original dataframe and create a new dataframe of the Manhattan data.

# In[15]:


manhattan_data = neighborhoods[neighborhoods['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_data.head()


# Let's get the geographical coordinates of Manhattan.

# In[16]:


address = 'Manhattan, NY'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# As we did with all of New York City, let's visualizat Manhattan the neighborhoods in it.

# In[17]:


# create map of Manhattan using latitude and longitude values
map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_manhattan)  
    
map_manhattan


# Next, we are going to start utilizing the Foursquare API to explore the neighborhoods and segment them.

# #### Define Foursquare Credentials and Version

# In[18]:


CLIENT_ID = 'your-client-ID' # your Foursquare ID
CLIENT_SECRET = 'your-client-secret' # your Foursquare Secret
VERSION = '201800924' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + "I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA")
print('CLIENT_SECRET:' + "RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0")


# #### Let's explore the first neighborhood in our dataframe.

# Get the neighborhood's name.

# In[19]:


manhattan_data.loc[0, 'Neighborhood']


# Get the neighborhood's latitude and longitude values.

# In[20]:


neighborhood_latitude = manhattan_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = manhattan_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = manhattan_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# #### Now, let's get the top 100 venues that are in Marble Hill within a radius of 500 meters.

# First, let's create the GET request URL. Name your URL **url**.

# In[21]:


LIMIT = 100 # limit of number of venues returned by Foursquare API


radius = 500 # define radius



url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.87655077879964, -73.91065965862981&radius=500&limit=100'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL




# Double-click __here__ for the solution.
# <!-- The correct answer is:
# LIMIT = 100 # limit of number of venues returned by Foursquare API
# -->
# 
# <!--
# radius = 500 # define radius
# -->
# 
# <!--
# \\ # create URL
# url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
#     CLIENT_ID, 
#     CLIENT_SECRET, 
#     VERSION, 
#     neighborhood_latitude, 
#     neighborhood_longitude, 
#     radius, 
#     LIMIT)
# url # display URL
# --> 

# Send the GET request and examine the resutls

# In[22]:


results = requests.get(url).json()


# From the Foursquare lab in the previous module, we know that all the information is in the *items* key. Before we proceed, let's borrow the **get_category_type** function from the Foursquare lab.

# In[23]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# Now we are ready to clean the json and structure it into a *pandas* dataframe.

# In[24]:


venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()


# And how many venues were returned by Foursquare?

# In[25]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# <a id='item2'></a>

# ## 2. Explore Neighborhoods in Manhattan

# #### Let's create a function to repeat the same process to all the neighborhoods in Manhattan

# In[26]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll={},{}&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# #### Now write the code to run the above function on each neighborhood and create a new dataframe called *manhattan_venues*.

# In[27]:


venues_list=[]
url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.876551,-73.91066&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
venues_list.append([(
            "Marble Hill", 
            40.876551, 
            -73.91066, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']


# In[28]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.715618,-73.994279&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
venues_list.append([(
            "Chinatown", 
            40.715618, 
            -73.994279, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']


# In[29]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.851903,-73.9369&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
venues_list.append([(
            "Washington Heights", 
            40.851903, 
            -73.9369, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']


# In[30]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.867684,-73.92121&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
venues_list.append([(
            "Hamilton Height", 
            40.823604, 
            -73.949688, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']


# In[31]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id=I3AZ3NT3IVE0Q5BIFADCORBUCJ5IKM0EA32YW5F5YZLL2NFA&client_secret=RUPI1WHIFUSAXGV0SXIMSSOSBLLKPIZ5OOMXNR42AGMPYEZ0&v=20180922&ll=40.823604,-73.949688&radius=500&limit=100'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()['response']['groups'][0]['items']
        
        # return only relevant information for each nearby venue
venues_list.append([(
            "Inwood", 
            40.867684, 
            -73.92121, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']


# In[33]:


manhattan_venues=nearby_venues


# In[35]:


manhattan_venues.head()


# #### Let's check the size of the resulting dataframe

# In[36]:


print(manhattan_venues.shape)
manhattan_venues.head()


# Let's check how many venues were returned for each neighborhood

# In[37]:


manhattan_venues.groupby('Neighborhood').count()


# #### Let's find out how many unique categories can be curated from all the returned venues

# In[38]:


print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))


# <a id='item3'></a>

# ## 3. Analyze Each Neighborhood

# In[39]:


# one hot encoding
manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['Neighborhood'] = manhattan_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_onehot.head()


# And let's examine the new dataframe size.

# In[40]:


manhattan_onehot.shape


# #### Next, let's group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[41]:


manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
manhattan_grouped


# #### Let's confirm the new size

# In[42]:


manhattan_grouped.shape


# #### Let's print each neighborhood along with the top 5 most common venues

# In[43]:


num_top_venues = 5

for hood in manhattan_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = manhattan_grouped[manhattan_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### Let's put that into a *pandas* dataframe

# First, let's write a function to sort the venues in descending order.

# In[44]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# Now let's create the new dataframe and display the top 10 venues for each neighborhood.

# In[45]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']

for ind in np.arange(manhattan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# <a id='item4'></a>

# ## 4. Cluster Neighborhoods

# Run *k*-means to cluster the neighborhood into 5 clusters.

# In[46]:


# set number of clusters
kclusters = 4

manhattan_grouped_clustering = manhattan_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_


# In[47]:


manhattan_grouped_clustering['Neighborhood']=manhattan_grouped['Neighborhood']
manhattan_grouped_clustering['Cluster Labels'] = kmeans.labels_
manhattan_grouped_clustering


# Let's create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood.

# Finally, let's visualize the resulting clusters

# In[48]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)
# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]
markers_colors = []
for lat, lon, poi in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):
    label = folium.Popup(str(poi), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        fill=True,
        fill_opacity=0.7).add_to(map_clusters)
map_clusters


# <a id='item5'></a>
