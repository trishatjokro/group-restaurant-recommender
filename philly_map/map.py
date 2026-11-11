import pandas as pd 
import folium as fl
from folium.plugins import MarkerCluster


# Remove restaurants without coordinates
def get_map(restaurants):
    cuisine_icons = {
        "Chinese": "🍜",
        "Japanese": "🍣",
        "Korean": "🥩",
        "Mexican": "🌮",
        "Italian": "🍕",
        "Indian": "🍛",
        "Thai": "🥢",
        "Vietnamese": "🍲",
        "American": "🍔",
        "Cafe": "☕",
        "Other": "🍽️"
    }

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
    icon_create_function = """
    function(cluster) {
        const markers = cluster.getAllChildMarkers();
        const counts = {};

        markers.forEach(function(marker) {
            const cuisine = marker.options.cuisine || "Other";
            counts[cuisine] = (counts[cuisine] || 0) + 1;
        });

        let mostCommon = "Other";
        let highestCount = 0;

        Object.entries(counts).forEach(function([cuisine, count]) {
            if (count > highestCount) {
                mostCommon = cuisine;
                highestCount = count;
            }
        });

        const cuisineIcons = {
            "Chinese": "🍜",
            "Japanese": "🍣",
            "Korean": "🥩",
            "Mexican": "🌮",
            "Italian": "🍕",
            "Indian": "🍛",
            "Thai": "🥢",
            "Vietnamese": "🍲",
            "American": "🍔",
            "Cafe": "☕",
            "Other": "🍽️"
        };

        const icon = cuisineIcons[mostCommon] || "🍽️";

        return L.divIcon({
            html: `
                <div style="
                    background-color: white;
                    width: 44px;
                    height: 44px;
                    border: 1px solid #555;
                    border-radius: 50%;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                ">
                    <span style="font-size: 20px; line-height: 20px;">
                        ${icon}
                    </span>
                    <span style="font-size: 11px; font-weight: bold;">
                        ${markers.length}
                    </span>
                </div>
            `,
            className: "custom-cluster-icon",
            iconSize: L.point(44, 44)
        });
    }
    """

    marker_cluster = MarkerCluster(icon_create_function=icon_create_function).add_to(restaurant_map)

    for _, restaurant in locations.iterrows():
        cuisine = restaurant["cuisine_group"]
        icon = cuisine_icons[cuisine]

        rating = restaurant["stars"]
        color = rating_colors.get(rating, "gray")

        marker_icon = fl.DivIcon(
            html=f"""
            <div style="
                background-color: {color};
                width: 32px;
                height: 32px;
                border: 2px solid white;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.5);
            ">
                {icon}
            </div>
            """
        )

        fl.Marker(
            location=[restaurant["latitude"], restaurant["longitude"]],
            tooltip=restaurant["name"],
            popup=fl.Popup(
                (
                    f'<div style="text-align: center;">'
                    f"<b>{restaurant['name']}</b><br>"
                    f"Cuisine: {cuisine}<br>"
                    f"Rating: {restaurant['stars']}<br>"
                    f"{restaurant['address']}"
                    f"</div>"
                ),
                min_width=150,
                max_width=150
            ),
            icon=marker_icon,
            cuisine=cuisine 
        ).add_to(marker_cluster)

    restaurant_map.save("restaurant_map.html")
    print("Map saved to restaurant_map.html")

#sort restaurants by categories 
def get_cuisine(categories):
    categories = str(categories).lower()

    if any(word in categories for word in ["chinese", "szechuan", "dim sum"]):
        return "Chinese"
    elif any(word in categories for word in ["japanese", "sushi", "ramen"]):
        return "Japanese"
    elif any(word in categories for word in ["korean"]):
        return "Korean"
    elif any(word in categories for word in ["mexican", "tacos", "burritos"]):
        return "Mexican"
    elif any(word in categories for word in ["italian", "pizza", "pasta"]):
        return "Italian"
    elif any(word in categories for word in ["indian", "pakistani"]):
        return "Indian"
    elif "thai" in categories:
        return "Thai"
    elif any(word in categories for word in ["vietnamese", "pho"]):
        return "Vietnamese"
    elif any(word in categories for word in ["burgers", "american"]):
        return "American"
    elif any(word in categories for word in ["coffee", "cafes", "tea"]):
        return "Cafe"
    else:
        return "Other"



if __name__ == "__main__":
    
    restaurants = pd.read_parquet("restaurants_clean.parquet")
    # print(restaurants.shape)
    # print(restaurants['categories'].head())
    # restaurants.info()
    restaurants["cuisine_group"] = restaurants["categories"].apply(get_cuisine)
    # print(restaurants[ ["name", "categories", "cuisine_group"] ].head(20))
    # print(restaurants["cuisine_group"].value_counts())
    # print(restaurants["stars"].head(10))
    get_map(restaurants)
    #python -m http.server 8000
    #open http://localhost:8000/restaurant_map.html in browser to view map
    