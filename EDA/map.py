import pandas as pd 
import folium as fl
from folium.plugins import MarkerCluster


# Remove restaurants without coordinates
def get_map(restaurants):
    rating_colors = {
    1.0: "darkred",
    1.5: "red",
    2.0: "lightred",
    2.5: "orange",
    3.0: "beige",
    3.5: "lightgreen",
    4.0: "green",
    4.5: "darkgreen",
    5.0: "blue"
}
    #drop restaurants without coordinates 
    locations = restaurants.dropna(subset=["latitude", "longitude"]).copy()

    # create a map centered around the restaurants
    restaurant_map = fl.Map(
        location=[locations["latitude"].mean(), locations["longitude"].mean()],
        zoom_start=10
    )

    # cluster nearby restaurants 
    marker_cluster = MarkerCluster().add_to(restaurant_map)

    for _, restaurant in locations.iterrows():
        rating = restaurant["stars"]
        color = rating_colors.get(rating, "gray")
        fl.Marker(
            location=[restaurant["latitude"], restaurant["longitude"]],
            tooltip=restaurant["name"],
            popup=fl.Popup(
                (
                    f'<div style="text-align: center;">'
                    f"<b>{restaurant['name']}</b><br>"
                    f"Rating: {restaurant['stars']}<br>"
                    f"{restaurant['address']}"
                    f"</div>"
                ),
                min_width=150,
                max_width=150
            ),
            icon=fl.Icon(color=color, icon="cutlery", prefix="fa")
        ).add_to(marker_cluster)

    restaurant_map.save("restaurant_map.html")
    print("Map saved to restaurant_map.html")




if __name__ == "__main__":
    restaurants = pd.read_parquet("data/output/restaurants_clean.parquet")
    print(restaurants.shape)
    # print(restaurants["stars"].head(10))
    get_map(restaurants)
    #python -m http.server 8000
    #open http://localhost:8000/restaurant_map.html in browser to view map
    