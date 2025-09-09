# For album recommendation based on popularity,
# 1.) I removed all songs below median popularity(because many non popular songs in an album might add up in total popularity which will give bad recommendation). 
# 2.) added popularity scores of each song in an album and sorted based on total popularity score.


import pandas as pd
df = pd.read_csv("c:/Users/Vishnu/Downloads/Compressed/rolling_stones_spotify.csv")

# Step 1: Filter songs above median popularity
median = df['popularity'].median()  # or use mean()
filtered_df = df[df['popularity'] >= median]

# Step 2: Group by album and sum popularity
album_popularity = filtered_df.groupby('album')['popularity'].sum().sort_values(ascending=False).head(2)

# Step 3: Plot
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x=album_popularity.values, y=album_popularity.index)
plt.xlabel("Total Popularity of Above-Median Songs")
plt.ylabel("Album")
plt.title("Top 2 Albums")
plt.show()

# Most popular albums are Honk(Deluxe) and Exile On Main Street (Deluxe Version).


# to get the features assosciated with most popular songs, 
# 1.) I took the most popular 25 percent songs and least popular 25 percent songs 
# 2.) took the difference of means of their features.
# 3. Higher the difference, higher contribution of that feature in popularity.

# top 25% - most popular
pop_threshold = df['popularity'].quantile(0.75)
most_popular = df[df['popularity'] >= pop_threshold]
features = ['danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo']

# Bottom 25% - least popular
least_popular = df[df['popularity'] <= df['popularity'].quantile(0.25)]

# Mean feature comparison
popular_mean = most_popular[features].mean()
unpopular_mean = least_popular[features].mean()

# Difference
diff = (popular_mean - unpopular_mean).sort_values(ascending=False)
print(diff)
# Features related to popularity are acousticness, danceability, valance and instrumentalness

# to get the shift in features for popularity over the years, 
# 1.) divide the data into two sets based on median year, old songs and new songs
# 2.) take top 25 percent of old songs and new songs.
# 3.) take difference of mean features to get the most differential features.

df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = df['release_date'].dt.year

# Median year to split
median_year = df['year'].median()
old_songs = df[df['year'] <= median_year]
new_songs = df[df['year'] > median_year]

old_top = old_songs[old_songs['popularity'] >= old_songs['popularity'].quantile(0.75)]
new_top = new_songs[new_songs['popularity'] >= new_songs['popularity'].quantile(0.75)]

features = ['danceability', 'energy', 'valence', 'acousticness',
            'instrumentalness', 'liveness', 'speechiness', 'tempo']

old_top_mean = old_top[features].mean()
new_top_mean = new_top[features].mean()

shift = (new_top_mean - old_top_mean).sort_values(ascending=False)
print(shift)
# new songs have increased popularity if it has high tempo, liveliness, energy and speechiness. when compared to old songs.