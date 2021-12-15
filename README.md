# **Property Rental Price Estimation in Paris**
Author: Waldy Setiono (waldysetiono@gmail.com)

**Background**: An online accomodation and property rental company based in South East Asia plans to expand its service area to Europe and the first city will be Paris. The company is now conducting research to determine the pricing strategy in order to compete with existing similar companies. There has been no standard method to determine how much a property owner should cost their customers so this company considers analyzing historical pricing data of one of its major competitors to predict its potential future service prices. This project aims to make a predictive model that can estimate property rental price based on facility, location, room capacity, and other related features.

**Data**: The data used in this project is from [insideairbnb.com](http://insideairbnb.com/).

## **Outline**

1. Data Preparation

2. Exploratory Data Analysis and Data Cleaning

3. Feature Engineering

4. Modeling and Evaluation

## **Data Preparation**

**Import packages and load data**


```python
# Import packages
import pandas as pd
import numpy as np
import requests
from zipfile import ZipFile
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
% pip install geopandas
import geopandas as gpd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score
import time
from keras import models, layers, optimizers, regularizers
```

    Requirement already satisfied: geopandas in /usr/local/lib/python3.7/dist-packages (0.10.2)
    Requirement already satisfied: pyproj>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from geopandas) (3.2.1)
    Requirement already satisfied: shapely>=1.6 in /usr/local/lib/python3.7/dist-packages (from geopandas) (1.7.1)
    Requirement already satisfied: fiona>=1.8 in /usr/local/lib/python3.7/dist-packages (from geopandas) (1.8.20)
    Requirement already satisfied: pandas>=0.25.0 in /usr/local/lib/python3.7/dist-packages (from geopandas) (1.1.5)
    Requirement already satisfied: click>=4.0 in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (7.1.2)
    Requirement already satisfied: cligj>=0.5 in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (0.7.2)
    Requirement already satisfied: six>=1.7 in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (1.15.0)
    Requirement already satisfied: attrs>=17 in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (21.2.0)
    Requirement already satisfied: munch in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (2.5.0)
    Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (2021.5.30)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (57.4.0)
    Requirement already satisfied: click-plugins>=1.0 in /usr/local/lib/python3.7/dist-packages (from fiona>=1.8->geopandas) (1.1.1)
    Requirement already satisfied: numpy>=1.15.4 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25.0->geopandas) (1.19.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25.0->geopandas) (2.8.2)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25.0->geopandas) (2018.9)
    


```python
r = requests.get("https://github.com/waldysetio/price-estimation/blob/main/data/listings.zip?raw=true")
files = ZipFile(BytesIO(r.content))
data = pd.read_csv(files.open("listings.csv"))
data
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>listing_url</th>
      <th>scrape_id</th>
      <th>last_scraped</th>
      <th>name</th>
      <th>description</th>
      <th>neighborhood_overview</th>
      <th>picture_url</th>
      <th>host_id</th>
      <th>host_url</th>
      <th>host_name</th>
      <th>host_since</th>
      <th>host_location</th>
      <th>host_about</th>
      <th>host_response_time</th>
      <th>host_response_rate</th>
      <th>host_acceptance_rate</th>
      <th>host_is_superhost</th>
      <th>host_thumbnail_url</th>
      <th>host_picture_url</th>
      <th>host_neighbourhood</th>
      <th>host_listings_count</th>
      <th>host_total_listings_count</th>
      <th>host_verifications</th>
      <th>host_has_profile_pic</th>
      <th>host_identity_verified</th>
      <th>neighbourhood</th>
      <th>neighbourhood_cleansed</th>
      <th>neighbourhood_group_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bathrooms_text</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>amenities</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>minimum_minimum_nights</th>
      <th>maximum_minimum_nights</th>
      <th>minimum_maximum_nights</th>
      <th>maximum_maximum_nights</th>
      <th>minimum_nights_avg_ntm</th>
      <th>maximum_nights_avg_ntm</th>
      <th>calendar_updated</th>
      <th>has_availability</th>
      <th>availability_30</th>
      <th>availability_60</th>
      <th>availability_90</th>
      <th>availability_365</th>
      <th>calendar_last_scraped</th>
      <th>number_of_reviews</th>
      <th>number_of_reviews_ltm</th>
      <th>number_of_reviews_l30d</th>
      <th>first_review</th>
      <th>last_review</th>
      <th>review_scores_rating</th>
      <th>review_scores_accuracy</th>
      <th>review_scores_cleanliness</th>
      <th>review_scores_checkin</th>
      <th>review_scores_communication</th>
      <th>review_scores_location</th>
      <th>review_scores_value</th>
      <th>license</th>
      <th>instant_bookable</th>
      <th>calculated_host_listings_count</th>
      <th>calculated_host_listings_count_entire_homes</th>
      <th>calculated_host_listings_count_private_rooms</th>
      <th>calculated_host_listings_count_shared_rooms</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5396</td>
      <td>https://www.airbnb.com/rooms/5396</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Explore the heart of old Paris</td>
      <td>Cozy, well-appointed and graciously designed s...</td>
      <td>You are within walking distance to the Louvre,...</td>
      <td>https://a0.muscache.com/pictures/52413/f9bf76f...</td>
      <td>7903</td>
      <td>https://www.airbnb.com/users/show/7903</td>
      <td>Borzou</td>
      <td>2009-02-14</td>
      <td>İstanbul, İstanbul, Turkey</td>
      <td>The flat is owned by journalists who spend a l...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>89%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/7903/profile_...</td>
      <td>https://a0.muscache.com/im/users/7903/profile_...</td>
      <td>Saint-Paul - Ile Saint-Louis</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'facebook', 'google', 'revi...</td>
      <td>t</td>
      <td>t</td>
      <td>Paris, Ile-de-France, France</td>
      <td>Hôtel-de-Ville</td>
      <td>NaN</td>
      <td>48.852470</td>
      <td>2.358350</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>["Hot water kettle", "Cooking basics", "Smoke ...</td>
      <td>$110.00</td>
      <td>2</td>
      <td>1125</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1125.0</td>
      <td>1125.0</td>
      <td>2.0</td>
      <td>1125.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>0</td>
      <td>3</td>
      <td>29</td>
      <td>29</td>
      <td>2021-09-11</td>
      <td>260</td>
      <td>35</td>
      <td>3</td>
      <td>2013-09-22</td>
      <td>2020-08-08</td>
      <td>4.51</td>
      <td>4.55</td>
      <td>4.47</td>
      <td>4.78</td>
      <td>4.82</td>
      <td>4.96</td>
      <td>4.53</td>
      <td>7510402838018</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7397</td>
      <td>https://www.airbnb.com/rooms/7397</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>MARAIS - 2ROOMS APT - 2/4 PEOPLE</td>
      <td>VERY CONVENIENT, WITH THE BEST LOCATION !&lt;br /...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/67928287/330b...</td>
      <td>2626</td>
      <td>https://www.airbnb.com/users/show/2626</td>
      <td>Franck</td>
      <td>2008-08-30</td>
      <td>Paris, Île-de-France, France</td>
      <td>I am a writer,51, author of novels, books of l...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>80%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/pictures/user/58f00...</td>
      <td>https://a0.muscache.com/im/pictures/user/58f00...</td>
      <td>Le Marais</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Hôtel-de-Ville</td>
      <td>NaN</td>
      <td>48.859090</td>
      <td>2.353150</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>["Cooking basics", "Smoke alarm", "Iron", "Ove...</td>
      <td>$100.00</td>
      <td>10</td>
      <td>130</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>130.0</td>
      <td>130.0</td>
      <td>10.0</td>
      <td>130.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>257</td>
      <td>2021-09-11</td>
      <td>278</td>
      <td>9</td>
      <td>2</td>
      <td>2011-08-11</td>
      <td>2021-08-18</td>
      <td>4.70</td>
      <td>4.79</td>
      <td>4.44</td>
      <td>4.91</td>
      <td>4.88</td>
      <td>4.92</td>
      <td>4.70</td>
      <td>7510400829623</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2.26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7964</td>
      <td>https://www.airbnb.com/rooms/7964</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Large &amp; sunny flat with balcony !</td>
      <td>Very large &amp; nice apartment all for you! &lt;br /...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/4471349/6fb3d...</td>
      <td>22155</td>
      <td>https://www.airbnb.com/users/show/22155</td>
      <td>Anaïs</td>
      <td>2009-06-18</td>
      <td>Paris, Île-de-France, France</td>
      <td>Hello ! \r\nOur apartment is great and I am su...</td>
      <td>within a day</td>
      <td>60%</td>
      <td>0%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/22155/profile...</td>
      <td>https://a0.muscache.com/im/users/22155/profile...</td>
      <td>Gare du Nord - Gare de I'Est</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Opéra</td>
      <td>NaN</td>
      <td>48.874170</td>
      <td>2.342450</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Essentials", "TV with standard cable", "Wifi...</td>
      <td>$130.00</td>
      <td>6</td>
      <td>365</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>365.0</td>
      <td>365.0</td>
      <td>6.0</td>
      <td>365.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>13</td>
      <td>43</td>
      <td>73</td>
      <td>348</td>
      <td>2021-09-11</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>2014-09-11</td>
      <td>2015-09-14</td>
      <td>4.80</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>7510903576564</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9359</td>
      <td>https://www.airbnb.com/rooms/9359</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Cozy, Central Paris: WALK or VELIB EVERYWHERE !</td>
      <td>Location! Location! Location! Just bring your ...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/c2965945-061f...</td>
      <td>28422</td>
      <td>https://www.airbnb.com/users/show/28422</td>
      <td>Bernadette</td>
      <td>2009-07-29</td>
      <td>New York, New York, United States</td>
      <td>I am a Native New Yorker (yes, I was born and ...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>20%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/users/28422/profile...</td>
      <td>https://a0.muscache.com/im/users/28422/profile...</td>
      <td>Châtelet - Les Halles - Beaubourg</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Louvre</td>
      <td>NaN</td>
      <td>48.860060</td>
      <td>2.348630</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Essentials", "Wifi", "Kitchen", "Hair dryer"...</td>
      <td>$75.00</td>
      <td>180</td>
      <td>365</td>
      <td>180.0</td>
      <td>180.0</td>
      <td>365.0</td>
      <td>365.0</td>
      <td>180.0</td>
      <td>365.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>28</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Available with a mobility lease only ("bail mo...</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9952</td>
      <td>https://www.airbnb.com/rooms/9952</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Paris petit coin douillet</td>
      <td>Je suis une dame retraitée, qui propose un agr...</td>
      <td>Vibrant neighborhood, full of bars, cafés, fre...</td>
      <td>https://a0.muscache.com/pictures/ae822d16-74d2...</td>
      <td>33534</td>
      <td>https://www.airbnb.com/users/show/33534</td>
      <td>Elisabeth</td>
      <td>2009-08-24</td>
      <td>Paris, Île-de-France, France</td>
      <td>Parisienne retraitée, dynamique et accueillant...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>100%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/pictures/user/4f775...</td>
      <td>https://a0.muscache.com/im/pictures/user/4f775...</td>
      <td>République</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'offlin...</td>
      <td>t</td>
      <td>t</td>
      <td>Paris, Ile-de-France, France</td>
      <td>Popincourt</td>
      <td>NaN</td>
      <td>48.863730</td>
      <td>2.370930</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["TV", "Cooking basics", "Smoke alarm", "Lugga...</td>
      <td>$80.00</td>
      <td>4</td>
      <td>31</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>31.0</td>
      <td>31.0</td>
      <td>4.0</td>
      <td>31.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>198</td>
      <td>2021-09-11</td>
      <td>31</td>
      <td>6</td>
      <td>1</td>
      <td>2016-08-04</td>
      <td>2021-06-23</td>
      <td>4.94</td>
      <td>4.97</td>
      <td>4.87</td>
      <td>5.00</td>
      <td>4.90</td>
      <td>4.90</td>
      <td>4.94</td>
      <td>7511101582862</td>
      <td>f</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>50128</th>
      <td>52162381</td>
      <td>https://www.airbnb.com/rooms/52162381</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Cosy Apartment 4P/1BD Canal St-Martin</td>
      <td>This very cute apartment will be perfect for a...</td>
      <td>This apartment is located in a very lively dis...</td>
      <td>https://a0.muscache.com/pictures/prohost-api/H...</td>
      <td>325819242</td>
      <td>https://www.airbnb.com/users/show/325819242</td>
      <td>Checkmyguest</td>
      <td>2020-01-09</td>
      <td>Paris, Île-de-France, France</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/3cfaf...</td>
      <td>https://a0.muscache.com/im/pictures/user/3cfaf...</td>
      <td>Pigalle - Saint-Georges</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>['email', 'phone']</td>
      <td>t</td>
      <td>t</td>
      <td>Paris, Île-de-France, France</td>
      <td>Entrepôt</td>
      <td>NaN</td>
      <td>48.874007</td>
      <td>2.363030</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>["Hot water kettle", "TV", "Cooking basics", "...</td>
      <td>$142.00</td>
      <td>1</td>
      <td>1125</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1125.0</td>
      <td>1125.0</td>
      <td>2.9</td>
      <td>1125.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>26</td>
      <td>56</td>
      <td>86</td>
      <td>361</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7511005130319</td>
      <td>t</td>
      <td>31</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50129</th>
      <td>52162674</td>
      <td>https://www.airbnb.com/rooms/52162674</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Tour Eiffel / Passy // magnifique appartement 4P</td>
      <td>Ce superbe appartement est situé dans le fameu...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/01ba90ae-6838...</td>
      <td>353064334</td>
      <td>https://www.airbnb.com/users/show/353064334</td>
      <td>Home Suite</td>
      <td>2020-07-02</td>
      <td>FR</td>
      <td>NaN</td>
      <td>within an hour</td>
      <td>75%</td>
      <td>99%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/31e4e...</td>
      <td>https://a0.muscache.com/im/pictures/user/31e4e...</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>['email', 'phone', 'offline_government_id', 's...</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Passy</td>
      <td>NaN</td>
      <td>48.853539</td>
      <td>2.281483</td>
      <td>Entire condominium (condo)</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>["Private entrance", "Essentials", "TV", "Wifi...</td>
      <td>$138.00</td>
      <td>1</td>
      <td>1125</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1125.0</td>
      <td>1.4</td>
      <td>705.4</td>
      <td>NaN</td>
      <td>t</td>
      <td>19</td>
      <td>49</td>
      <td>79</td>
      <td>354</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7511605709975</td>
      <td>t</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50130</th>
      <td>52163316</td>
      <td>https://www.airbnb.com/rooms/52163316</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>❉Comfortable studio for 2p - Paris 3❉</td>
      <td>Come and discover my charming studio. This apa...</td>
      <td>PLACE DES VOSGES &lt;br /&gt;It is the jewel of the ...</td>
      <td>https://a0.muscache.com/pictures/miso/Hosting-...</td>
      <td>50502817</td>
      <td>https://www.airbnb.com/users/show/50502817</td>
      <td>WeHost</td>
      <td>2015-12-04</td>
      <td>Paris, Île-de-France, France</td>
      <td>Conciergerie Airbnb</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/a6ba3...</td>
      <td>https://a0.muscache.com/im/pictures/user/a6ba3...</td>
      <td>Commerce - Dupleix</td>
      <td>175.0</td>
      <td>175.0</td>
      <td>['email', 'phone', 'reviews', 'jumio', 'govern...</td>
      <td>t</td>
      <td>f</td>
      <td>Paris, Île-de-France, France</td>
      <td>Temple</td>
      <td>NaN</td>
      <td>48.861777</td>
      <td>2.364823</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Hot water kettle", "Refrigerator", "Shower g...</td>
      <td>$61.00</td>
      <td>3</td>
      <td>1125</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1125.0</td>
      <td>1125.0</td>
      <td>3.0</td>
      <td>1125.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>21</td>
      <td>51</td>
      <td>81</td>
      <td>261</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7510305662032</td>
      <td>t</td>
      <td>61</td>
      <td>54</td>
      <td>4</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50131</th>
      <td>52165011</td>
      <td>https://www.airbnb.com/rooms/52165011</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Cosy flat between Bastille and République in P...</td>
      <td>Located at the heart of Paris, between some of...</td>
      <td>Staying in the 11th arrondissement will enable...</td>
      <td>https://a0.muscache.com/pictures/prohost-api/H...</td>
      <td>125797498</td>
      <td>https://www.airbnb.com/users/show/125797498</td>
      <td>Welkeys</td>
      <td>2017-04-14</td>
      <td>Paris, Île-de-France, France</td>
      <td>Bienvenue Chez Vous !\r\n\r\nWelkeys est une s...</td>
      <td>within an hour</td>
      <td>92%</td>
      <td>98%</td>
      <td>f</td>
      <td>https://a0.muscache.com/im/pictures/user/76dfd...</td>
      <td>https://a0.muscache.com/im/pictures/user/76dfd...</td>
      <td>2nd Arrondissement</td>
      <td>138.0</td>
      <td>138.0</td>
      <td>['email', 'phone', 'google', 'reviews', 'offli...</td>
      <td>t</td>
      <td>t</td>
      <td>Paris, Île-de-France, France</td>
      <td>Popincourt</td>
      <td>NaN</td>
      <td>48.857923</td>
      <td>2.375364</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>["Refrigerator", "Essentials", "Bed linens", "...</td>
      <td>$100.00</td>
      <td>2</td>
      <td>1125</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1125.0</td>
      <td>2.0</td>
      <td>33.7</td>
      <td>NaN</td>
      <td>t</td>
      <td>25</td>
      <td>55</td>
      <td>85</td>
      <td>287</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7511100347794</td>
      <td>t</td>
      <td>33</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50132</th>
      <td>52166001</td>
      <td>https://www.airbnb.com/rooms/52166001</td>
      <td>20210909211521</td>
      <td>2021-09-11</td>
      <td>Bail Mobilité Montmartre</td>
      <td>I'm renting a charming rooftop flat in montmar...</td>
      <td>NaN</td>
      <td>https://a0.muscache.com/pictures/50262813/7d17...</td>
      <td>18045582</td>
      <td>https://www.airbnb.com/users/show/18045582</td>
      <td>Betty</td>
      <td>2014-07-13</td>
      <td>Paris, Île-de-France, France</td>
      <td>bonjour ,\r\n je suis psychologue et formatric...</td>
      <td>within an hour</td>
      <td>100%</td>
      <td>100%</td>
      <td>t</td>
      <td>https://a0.muscache.com/im/users/18045582/prof...</td>
      <td>https://a0.muscache.com/im/users/18045582/prof...</td>
      <td>Pigalle - Saint-Georges</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>['email', 'phone', 'reviews', 'offline_governm...</td>
      <td>t</td>
      <td>t</td>
      <td>NaN</td>
      <td>Opéra</td>
      <td>NaN</td>
      <td>48.880568</td>
      <td>2.335568</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>NaN</td>
      <td>1 bath</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>["Cooking basics", "Luggage dropoff allowed", ...</td>
      <td>$69.00</td>
      <td>30</td>
      <td>1125</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>1125.0</td>
      <td>1125.0</td>
      <td>30.0</td>
      <td>1125.0</td>
      <td>NaN</td>
      <td>t</td>
      <td>0</td>
      <td>17</td>
      <td>47</td>
      <td>126</td>
      <td>2021-09-11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Available with a mobility lease only ("bail mo...</td>
      <td>f</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>50133 rows × 74 columns</p>
</div>



**Check boolean and numerical categories**


```python
# Plotting the distribution of numerical and boolean categories
data.hist(figsize=(25,25));
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_10_0.png)


**Drop unrelated columns**

This project is not going to implement sentiment analysis or text processing to make a predictive model, maybe in the future. Hence, for now columns containing reviews, summary, rules, and so on will be dropped. Columns that have just single unique value will be eliminated as well. Other columns that do not seem to be related to price will also be removed for simplicity reason but we might use them later if necessary.


```python
# Drop unused columns 
data = data.drop(["listing_url", "scrape_id", "last_scraped", "name", 
           "description", "neighborhood_overview", "picture_url", "host_id", 
           "host_url", "host_name", "host_location", "host_about", "host_thumbnail_url",
           "host_picture_url", "host_thumbnail_url", "host_picture_url", "host_verifications",
           "calendar_last_scraped", "number_of_reviews_l30d", "number_of_reviews_ltm",
           "first_review", "last_review", "license", "reviews_per_month", "host_neighbourhood",
           "host_listings_count", "host_has_profile_pic",
           "host_acceptance_rate", "host_total_listings_count", 
           "host_identity_verified", "neighbourhood", 
           "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights", 
           "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm", 
           "amenities", "has_availability", "availability_30", "availability_60", 
           "availability_90", "availability_365", "calculated_host_listings_count",
           "calculated_host_listings_count_entire_homes", "calculated_host_listings_count_private_rooms",
           "calculated_host_listings_count_shared_rooms"
           ], axis=1)
```

**Missing values**


```python
# Calculate missing values
missing_data_pecentage = data.isna().sum()/len(data.index)*100

# Plot missing values
missing_data_pecentage.plot(kind="bar", color="darkorange", figsize=(10,7))
plt.xlabel("Variables")
plt.ylabel("Missing values (%)")
plt.show()
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_15_0.png)


We are going to drop columns with significant missing values. Row(s) in the "id" column that contains NaN will also be dropped because we can't subtitute data in "id" column with alternative data such as median, mean, mode, or else.


```python
# Drop columns with significant missing data
data = data.drop(["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", 
                  "review_scores_checkin", "review_scores_communication", "calendar_updated",
                  "review_scores_location", "review_scores_value", "neighbourhood_group_cleansed", 
                  "host_response_time", "host_response_rate"],
                  axis=1)
```


```python
# Find row in which "id" that contains NaN value(s)
data[data["id"].isnull()]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_since</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bathrooms_text</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
      <th>instant_bookable</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



It seems there is no missing values in "id" column.

**Check duplicates**


```python
data[data.duplicated()]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_since</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bathrooms_text</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
      <th>instant_bookable</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



**Descriptive statistics**


```python
# Print basic statistics of the data
data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>minimum_nights</th>
      <th>maximum_nights</th>
      <th>number_of_reviews</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.013300e+04</td>
      <td>50133.000000</td>
      <td>50133.000000</td>
      <td>50133.000000</td>
      <td>0.0</td>
      <td>40700.000000</td>
      <td>49542.000000</td>
      <td>50133.000000</td>
      <td>5.013300e+04</td>
      <td>50133.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.594199e+07</td>
      <td>48.863973</td>
      <td>2.344874</td>
      <td>3.053019</td>
      <td>NaN</td>
      <td>1.374103</td>
      <td>1.689092</td>
      <td>112.930226</td>
      <td>9.999625e+02</td>
      <td>20.834600</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.566613e+07</td>
      <td>0.018182</td>
      <td>0.033154</td>
      <td>1.637042</td>
      <td>NaN</td>
      <td>1.017934</td>
      <td>1.383799</td>
      <td>170.370611</td>
      <td>4.466482e+04</td>
      <td>44.955323</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.396000e+03</td>
      <td>48.813080</td>
      <td>2.223870</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000e+00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.214614e+07</td>
      <td>48.850810</td>
      <td>2.324160</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>2.400000e+02</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.544425e+07</td>
      <td>48.865280</td>
      <td>2.347970</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>1.125000e+03</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.004825e+07</td>
      <td>48.878450</td>
      <td>2.369280</td>
      <td>4.000000</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>365.000000</td>
      <td>1.125000e+03</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.216600e+07</td>
      <td>48.905680</td>
      <td>2.473190</td>
      <td>16.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>90.000000</td>
      <td>9999.000000</td>
      <td>1.000000e+07</td>
      <td>1596.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the statistics including columns with object data type
data.describe(include=['object'])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>host_since</th>
      <th>host_is_superhost</th>
      <th>neighbourhood_cleansed</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>bathrooms_text</th>
      <th>price</th>
      <th>instant_bookable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50099</td>
      <td>50099</td>
      <td>50133</td>
      <td>50133</td>
      <td>50133</td>
      <td>50030</td>
      <td>50133</td>
      <td>50133</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>3965</td>
      <td>2</td>
      <td>20</td>
      <td>66</td>
      <td>4</td>
      <td>31</td>
      <td>856</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>2019-09-02</td>
      <td>f</td>
      <td>Buttes-Montmartre</td>
      <td>Entire rental unit</td>
      <td>Entire home/apt</td>
      <td>1 bath</td>
      <td>$80.00</td>
      <td>f</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>504</td>
      <td>43197</td>
      <td>5355</td>
      <td>38191</td>
      <td>41329</td>
      <td>37301</td>
      <td>2133</td>
      <td>34160</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print data types of each column
data.dtypes
```




    id                          int64
    host_since                 object
    host_is_superhost          object
    neighbourhood_cleansed     object
    latitude                  float64
    longitude                 float64
    property_type              object
    room_type                  object
    accommodates                int64
    bathrooms                 float64
    bathrooms_text             object
    bedrooms                  float64
    beds                      float64
    price                      object
    minimum_nights              int64
    maximum_nights              int64
    number_of_reviews           int64
    instant_bookable           object
    dtype: object



###**Inspect columns**

**longitude and latitude**

We will create a new dataframe consisting longitude and latitude then drop these two variables from the main dataframe later.


```python
# Make a dataframe from longitude and latitude
long_lat = data[['longitude', 'latitude']]
```


```python
# Drop longitude and latitude from main dataframe
data = data.drop(['longitude', 'latitude'], axis=1)
```

**host_since**

We will create a new feature called "active_days" by substracting the last_scraped date to host_since.


```python
# Convert to datetime
data.host_since = pd.to_datetime(data.host_since) 

# Calculate the number of days
data['active_days'] = (datetime(2019, 4, 9) - data.host_since).astype('timedelta64[D]')

# Print mean and median
print("Mean days as host:", round(data['active_days'].mean(),0))
print("Median days as host:", data['active_days'].median())

# Replace null values with the median
data.active_days.fillna(data.active_days.median(), inplace=True)
```

    Mean days as host: 1203.0
    Median days as host: 1356.0
    

**room_type and property_type**


```python
# Print categories in room_type
data.room_type.value_counts()
```




    Entire home/apt    41329
    Private room        7250
    Hotel room          1209
    Shared room          345
    Name: room_type, dtype: int64




```python
# Print categories in property_type
data.property_type.value_counts()
```




    Entire rental unit             38191
    Private room in rental unit     5053
    Room in boutique hotel          1714
    Entire condominium (condo)      1519
    Entire loft                      875
                                   ...  
    Private room in chalet             1
    Dome house                         1
    Entire bed and breakfast           1
    Entire cottage                     1
    Barn                               1
    Name: property_type, Length: 66, dtype: int64



Since room_type and property_type have similar categories, we will use only room type, drop the property_type, and categorize them to three labels which are "entire unit", "private room", and "other".


```python
# Replacing other categories with 'other'
data.loc[~data.property_type.isin(['Entire home/apt', 'Private room']), 'room_type'] = 'Other'
```


```python
# Drop property_type
data = data.drop(["property_type"], axis=1)
```

**bathrooms, bedrooms and beds**


```python
# Print categories in bathrooms
data.bathrooms.unique()
```




    array([nan])



Drop "bathrooms" since it only contains NaN values.


```python
# Drop "bathrooms"
data = data.drop(["bathrooms"], axis=1)
```

Let's change categorical data of bathrooms_text to numerical.


```python
# Print categories in bathrooms_text
data.bathrooms_text.unique()
```




    array(['1 bath', '1 private bath', '1.5 baths', '2 baths',
           '1 shared bath', '1.5 shared baths', nan, '2.5 baths',
           '2 shared baths', '4 baths', 'Half-bath', '3 baths', '3.5 baths',
           '4.5 baths', 'Shared half-bath', '7 shared baths', '0 baths',
           '0 shared baths', '5 baths', 'Private half-bath',
           '2.5 shared baths', '6.5 shared baths', '6 baths', '8 baths',
           '3 shared baths', '7 baths', '50 baths', '5.5 baths', '6.5 baths',
           '10 baths', '29 baths', '23 baths'], dtype=object)




```python
# Replace bathrooms_text data to numerical
data.bathrooms_text.replace({
    '1 bath': 1,
    '1 private bath': 1,
    '1.5 baths': 1,
    '2 baths': 2,
    '1 shared bath': 1,
    '1.5 shared baths': 1,
    '2.5 baths': 2,
    '2 shared baths': 2,
    '4 baths': 4,
    'Half-bath': 1,
    '3 baths': 3,
    '3.5 baths': 3,
    '4.5 baths': 4,
    '5.5 baths': 5,
    '6.5 baths': 6,
    'Shared half-bath': 1,
    '7 shared baths': 7,
    '0 baths': 0,
    '0 shared baths': 0,
    '5 baths': 5,
    'Private half-bath': 1,
    '2.5 shared baths': 2,
    '6.5 shared baths': 6,
    '6 baths': 6,
    '8 baths': 8,
    '3 shared baths': 3,
    '7 baths': 7, 
    '10 baths': 10,
    '23 baths': 23,
    '29 baths': 29,
    '50 baths': 50,

    }, inplace=True)
```


```python
# Replace null values with the median
data.bathrooms_text.fillna(data.bathrooms_text.median(), inplace=True)
```


```python
# Change data type to integer
data.bathrooms_text = data.bathrooms_text.astype('int32')
```


```python
# Change column name
data = data.rename(columns = {'bathrooms_text':'bathrooms'})
```

Change NaN values in beds to median.


```python
data.beds.unique()
```




    array([ 1.,  2.,  3.,  0.,  4.,  5.,  6.,  8., nan,  9.,  7., 11., 12.,
           16., 18., 10., 79., 77., 90., 83., 85., 13., 14., 40.])




```python
# Replace null values with the median
data.beds.fillna(data.beds.median(), inplace=True)
```


```python
# Change data type to integer
data.beds = data.beds.astype('int32')
```

**price**

We will drop the currency sign from price strings and change them to integer.


```python
# Format price
data.price = data.price.str[1:-3]
data.price = data.price.str.replace(",", "")
data.price = data.price.astype('int32')
data.price
```




    0        110
    1        100
    2        130
    3         75
    4         80
            ... 
    50128    142
    50129    138
    50130     61
    50131    100
    50132     69
    Name: price, Length: 50133, dtype: int32



**host_is_superhost**


```python
# Print value counts
print(data.host_is_superhost.unique())
print(data.host_is_superhost.value_counts())
```

    ['f' 't' nan]
    f    43197
    t     6902
    Name: host_is_superhost, dtype: int64
    


```python
# Replace binary categorical data to 0 and 1
data.host_is_superhost.replace({'f': 0, 't': 1}, inplace=True)
```


```python
# Replace null values with the median
data.host_is_superhost.fillna(data.host_is_superhost.median(), inplace=True)
```


```python
# Change data type to integer
data.host_is_superhost = data.host_is_superhost.astype('int32')
```

**instant_bookable**


```python
# Print value counts
print(data.instant_bookable.unique())
print(data.instant_bookable.value_counts())
```

    ['f' 't']
    f    34160
    t    15973
    Name: instant_bookable, dtype: int64
    


```python
# Replace binary categorical data to 0 and 1
data.instant_bookable.replace({'f': 0, 't': 1}, inplace=True)
```

##**Exploratory Data Analysis**

**Trend of Service Adoption**

Let's see the trend of hosts joining the service. We can see that the number kept increasing since 2009 until the peak in 2015 and it has been relatively slowing down until today with significant increase in 2019.


```python
# Create dataframes for time series analysis
ts_host_since = pd.DataFrame(data.set_index('host_since').resample('MS').size())
```


```python
# Rename columns
ts_host_since = ts_host_since.rename(columns={0: 'hosts'})
ts_host_since.index.rename('month', inplace=True)
```


```python
def decompose_time_series(df, title=''):
    """
    Plots the original time series and its decomposition into trend, seasonal and residual.
    """
    # Decomposing the time series
    decomposition = seasonal_decompose(df)
    
    # Getting the trend, seasonality and noise
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    # Plotting the original time series and the decomposition
    plt.figure(figsize=(12,10))
    plt.suptitle(title, fontsize=12, y=1)
    plt.subplots_adjust(top=0.80)
    plt.subplot(411)
    plt.plot(df, label='Original')
    plt.legend(loc='upper left')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='upper left')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality')
    plt.legend(loc='upper left')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='upper left')
    plt.tight_layout()
```


```python
# Call function of trend time series
decompose_time_series(ts_host_since, title='Number of hosts joining the service each month')
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_71_0.png)



```python
# Drop host_since as it is no longer needed
data.drop(['host_since'], axis=1, inplace=True)
```

**Price Distribution**


```python
# Print max and min of price
print(f"Nightly prices range from ${min(data.price)} to ${max(data.price)}.")
```

    Nightly prices range from $0 to $11600.
    


```python
# Plot the distribution of prices
plt.figure(figsize=(20,4))
data.price.hist(bins=100, range=(0,1000))
plt.margins(x=0)
plt.axvline(200, color='orange', linestyle='--')
plt.title("Nightly prices in Paris up to $1000", fontsize=16)
plt.xlabel("Price ($)")
plt.ylabel("Number of listings")
plt.show()

```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_75_0.png)



```python
# Distribution of prices from £200 upwards
plt.figure(figsize=(20,4))
data.price.hist(bins=100, range=(200, max(data.price)))
plt.margins(x=0)
plt.axvline(500, color='orange', linestyle='--')
plt.axvline(1000, color='red', linestyle='--')
plt.title("Nightly prices in Paris of more than $200", fontsize=16)
plt.xlabel("Price ($)")
plt.ylabel("Number of listings")
plt.show()
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_76_0.png)



```python
# Replacing values under $10 with $10
data.loc[data.price <= 10, 'price'] = 10

# Replacing values over $1000 with $1000
data.loc[data.price >= 1000, 'price'] = 1000
```

**Capacity**


```python
# Plot number of people that can be accomodated based on median price
plt.figure(figsize=(10,5))
data.groupby('accommodates').price.median().plot(kind='bar')
plt.title('Median price of places accommodating different number of guests', fontsize=14)
plt.xlabel('Number of guests accommodated', fontsize=13)
plt.ylabel('Median price ($)', fontsize=13)
plt.xticks(rotation=0)
plt.xlim(left=0.5)
plt.show()
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_79_0.png)


**Neighborhood**

We are going to see the distribution of price and the number of properties based on location.


```python
# Renaming the neighbourhood column
data.rename(columns={'neighbourhood_cleansed': 'area'}, inplace=True)

# Importing the London borough boundary GeoJSON file as a dataframe in geopandas
map_df = gpd.read_file('https://raw.githubusercontent.com/waldysetio/price-estimation/main/data/neighbourhoods.geojson')
map_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>neighbourhood</th>
      <th>neighbourhood_group</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Batignolles-Monceau</td>
      <td>None</td>
      <td>MULTIPOLYGON (((2.29517 48.87396, 2.29504 48.8...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Palais-Bourbon</td>
      <td>None</td>
      <td>MULTIPOLYGON (((2.32090 48.86306, 2.32094 48.8...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Buttes-Chaumont</td>
      <td>None</td>
      <td>MULTIPOLYGON (((2.38943 48.90122, 2.39014 48.9...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Opéra</td>
      <td>None</td>
      <td>MULTIPOLYGON (((2.33978 48.88203, 2.33982 48.8...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Entrepôt</td>
      <td>None</td>
      <td>MULTIPOLYGON (((2.36469 48.88437, 2.36485 48.8...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Dropping the empty column
map_df.drop('neighbourhood_group', axis=1, inplace=True)

# Creating a dataframe of listing counts and median price by borough
area_df = pd.DataFrame(data.groupby('area').size())
area_df.rename(columns={0: 'number_of_listings'}, inplace=True)
area_df['median_price'] = data.groupby('area').price.median().values

# Joining the dataframes
area_map_df = map_df.set_index('neighbourhood').join(area_df)
```


```python
# Plotting the number of listings in each area
fig1, ax1 = plt.subplots(1, figsize=(15, 6))
area_map_df.plot(column='number_of_listings', cmap='Blues', ax=ax1)
ax1.axis('off')
ax1.set_title('Number of listings in each Paris area', fontsize=14)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=9000))
sm._A = [] # Creates an empty array for the data range
cbar = fig1.colorbar(sm)
plt.show()

# Plotting the median price of listings in each area
fig2, ax2 = plt.subplots(1, figsize=(17, 6))
area_map_df.plot(column='median_price', cmap='Blues', ax=ax2)
ax2.axis('off')
ax2.set_title('Median price of listings in each Paris area', fontsize=14)
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=min(area_map_df.median_price), vmax=max(area_map_df.median_price)))
sm._A = [] # Create an empty array for the data range
cbar = fig2.colorbar(sm)
plt.show()
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_84_0.png)



![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_84_1.png)


**Boolean Features**


```python
def binary_count_and_price_plot(col, figsize=(8,3)):
    """
    Plots a simple bar chart of the counts of true and false categories in the column specified,
    next to a bar chart of the median price for each category.
    A figure size can optionally be specified.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(col, fontsize=16, y=1)
    plt.subplots_adjust(top=0.80) # So that the suptitle does not overlap with the ax plot titles
    
    data.groupby(col).size().plot(kind='bar', ax=ax1, color=['firebrick', 'seagreen'])
    ax1.set_xticklabels(labels=['false', 'true'], rotation=0)
    ax1.set_title('Category count')
    ax1.set_xlabel('')
    
    data.groupby(col).price.median().plot(kind='bar', ax=ax2, color=['firebrick', 'seagreen'])
    ax2.set_xticklabels(labels=['false', 'true'], rotation=0)
    ax2.set_title('Median price ($)')
    ax2.set_xlabel('')
    
    plt.show()
```

**Superhosts**


```python
# Correlation between price and whether host is super host
binary_count_and_price_plot('host_is_superhost')
print(data.host_is_superhost.value_counts(normalize=True))
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_88_0.png)


    0    0.862326
    1    0.137674
    Name: host_is_superhost, dtype: float64
    

**Instant booking**


```python
# Correlation between price and instant bookable
binary_count_and_price_plot('instant_bookable')
print(data.instant_bookable.value_counts(normalize=True))
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_90_0.png)


    0    0.681388
    1    0.318612
    Name: instant_bookable, dtype: float64
    

##**Feature Engineering**

**One-hot Encoding**


```python
# Make dummy variables
transformed_data = pd.get_dummies(data)
```

**Adressing Multicolinearity**


```python
def multi_collinearity_heatmap(df, figsize=(11,9)):
    
    """
    Creates a heatmap of correlations between features in the dataframe. A figure size can optionally be set.
    """
    
    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = df.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax=corr[corr != 1.0].max().max());
```


```python
# Plot correlation of transformed_data
multi_collinearity_heatmap(transformed_data, figsize=(20,20))
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_96_0.png)


There seems to be strong relationship between room_type_other with room_type_Private and bedrooms with beds. We will remove one of the two for these two pairs. 


```python
# Drop collinear features
to_drop = ['beds',
           'bedrooms',
           'room_type_Other', 
           'room_type_Private room']
to_drop.extend(list(transformed_data.columns[transformed_data.columns.str.endswith('nan')]))

transformed_data.drop(to_drop, axis=1, inplace=True)
```


```python
# See the multicollinearity without area feature
multi_collinearity_heatmap(transformed_data.drop(list(transformed_data.columns[transformed_data.columns.str.startswith('area')]), axis=1), figsize=(25,22))
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_99_0.png)


**Address Skewed Data**


```python
numerical_columns = ['accommodates', 'bathrooms', 'active_days', 'maximum_nights', 'minimum_nights', 'number_of_reviews', 'price']
```


```python
transformed_data[numerical_columns].hist(figsize=(10,11));
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_102_0.png)



```python
# Log transforming columns
numerical_columns = [i for i in numerical_columns if i not in ['host_days_active']] # Remove items not to be transformed

for col in numerical_columns:
    transformed_data[col] = transformed_data[col].astype('float64').replace(0.0, 0.01) # Replace 0s with 0.01
    transformed_data[col] = np.log(transformed_data[col])
```

    /usr/local/lib/python3.7/dist-packages/pandas/core/series.py:726: RuntimeWarning: invalid value encountered in log
      result = getattr(ufunc, method)(*inputs, **kwargs)
    


```python
transformed_data[numerical_columns].hist(figsize=(10,11));
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_104_0.png)



```python
# Separating X and y
X = transformed_data.drop('price', axis=1)
y = transformed_data.price

# Scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))
```

##**Modeling and Evaluation**


```python
# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
```

**XGB**


```python
# Modeling
xgb_reg_start = time.time()

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)
training_preds_xgb_reg = xgb_reg.predict(X_train)
val_preds_xgb_reg = xgb_reg.predict(X_test)

xgb_reg_end = time.time()

print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
print("\nTraining MSE:", round(mean_squared_error(y_train, training_preds_xgb_reg),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_preds_xgb_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_xgb_reg),4))
```

    [14:52:27] WARNING: /workspace/src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.
    Time taken to run: 0.1 minutes
    
    Training MSE: 0.1971
    Validation MSE: 0.2054
    
    Training r2: 0.5501
    Validation r2: 0.5187
    


```python
# Calculate weights
ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)
ft_weights_xgb_reg.sort_values('weight', inplace=True)
ft_weights_xgb_reg
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>room_type_Entire home/apt</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>area_Vaugirard</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>area_Entrepôt</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>area_Batignolles-Monceau</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>area_Observatoire</th>
      <td>0.008662</td>
    </tr>
    <tr>
      <th>maximum_nights</th>
      <td>0.010759</td>
    </tr>
    <tr>
      <th>area_Panthéon</th>
      <td>0.010760</td>
    </tr>
    <tr>
      <th>id</th>
      <td>0.011895</td>
    </tr>
    <tr>
      <th>active_days</th>
      <td>0.012367</td>
    </tr>
    <tr>
      <th>area_Opéra</th>
      <td>0.013760</td>
    </tr>
    <tr>
      <th>area_Bourse</th>
      <td>0.015044</td>
    </tr>
    <tr>
      <th>area_Popincourt</th>
      <td>0.018647</td>
    </tr>
    <tr>
      <th>area_Passy</th>
      <td>0.019185</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>0.020651</td>
    </tr>
    <tr>
      <th>area_Reuilly</th>
      <td>0.020975</td>
    </tr>
    <tr>
      <th>area_Temple</th>
      <td>0.023728</td>
    </tr>
    <tr>
      <th>area_Gobelins</th>
      <td>0.025696</td>
    </tr>
    <tr>
      <th>area_Palais-Bourbon</th>
      <td>0.025961</td>
    </tr>
    <tr>
      <th>area_Hôtel-de-Ville</th>
      <td>0.029520</td>
    </tr>
    <tr>
      <th>host_is_superhost</th>
      <td>0.029991</td>
    </tr>
    <tr>
      <th>area_Louvre</th>
      <td>0.030567</td>
    </tr>
    <tr>
      <th>area_Luxembourg</th>
      <td>0.032430</td>
    </tr>
    <tr>
      <th>minimum_nights</th>
      <td>0.032503</td>
    </tr>
    <tr>
      <th>area_Élysée</th>
      <td>0.043453</td>
    </tr>
    <tr>
      <th>area_Buttes-Chaumont</th>
      <td>0.048962</td>
    </tr>
    <tr>
      <th>area_Ménilmontant</th>
      <td>0.053697</td>
    </tr>
    <tr>
      <th>area_Buttes-Montmartre</th>
      <td>0.062794</td>
    </tr>
    <tr>
      <th>instant_bookable</th>
      <td>0.077156</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.137204</td>
    </tr>
    <tr>
      <th>accommodates</th>
      <td>0.183633</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plotting feature importances
plt.figure(figsize=(8,20))
plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center') 
plt.title("Feature importances in the XGBoost model", fontsize=14)
plt.xlabel("Feature importance")
plt.margins(y=0.01)
plt.show()
```


![png](https://github.com/waldysetio/price-estimation/blob/main/images/output_111_0.png)


**Note: This project is still ongoing. Feature engineering and model development will be improved to increase metrics of the model's quality.**
