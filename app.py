from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load dataset
df = pd.read_csv("dataset.csv")

# Select relevant features for recommendation
features = ["Danceability", "Energy", "Valence", "Tempo", "Duration_min"]

# Normalize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Train a Nearest Neighbors model
nn_model = NearestNeighbors(n_neighbors=5, metric="euclidean")
nn_model.fit(df_scaled)

def get_recommendations(song_name):
    """Returns top 5 similar songs based on features"""
    song = df[df["Track"].str.lower() == song_name.lower()]
    if song.empty:
        return None

    song_index = song.index[0]
    song_features = df_scaled[song_index].reshape(1, -1)

    distances, indices = nn_model.kneighbors(song_features)
    recommended_songs = df.iloc[indices[0][1:], :]

    return recommended_songs[["Artist", "Track", "Album", "Title"]].to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        song_name = request.form["song_name"]
        recommendations = get_recommendations(song_name)

        if recommendations is None:
            return render_template("result.html", error="Song not found!", song_name=song_name)

        return render_template("result.html", song_name=song_name, recommendations=recommendations)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)