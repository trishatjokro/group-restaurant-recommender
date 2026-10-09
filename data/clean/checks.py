# import os
# print(os.getcwd())

import pandas as pd
from rapidfuzz import process,fuzz

restaurants_all = pd.read_parquet("../restaurants_all.parquet")
restaurants_sample = pd.read_parquet("../restaurants_sample.parquet")
reviews_sample = pd.read_parquet("../reviews_sample.parquet")

def check_stars(df):
    print("Missing stars:", df["stars"].isna().sum())
    invalid = ~df["stars"].between(1, 5)|(df["stars"]%0.5!=0)
    print("Invalid stars:", invalid.sum())

def check_date(df):
    #missing dates
    missing_sum=df["date"].isna().sum()
    missing=df.loc[df["date"].isna(),["review_id","business_id","date"]]
    print("Missing dates:", missing_sum)
    if len(missing)>0:
        print("Missing dates:",missing)

    #check format
    non_missing = df.loc[df["date"].notna(), "date"] #non-missing dates
    correct_format=pd.to_datetime(non_missing,format="%Y-%m-%d %H:%M:%S",errors="coerce") #check if correct_format, if not, NaT, Series
    invalid=(correct_format.isna() | (correct_format>pd.Timestamp.now())) #.isna() for data tht cant be converted (NaT), check future dates
    print("Invalid dates:", invalid.sum())
    if invalid.any(): #.any() return true is any invalid dates
        print("Invalid dates:\n",df.loc[
        correct_format.index[invalid],
        ["review_id", "business_id", "date"]])

def check_cities(df):
    print("Missing cities:", df["city"].isna().sum())

def clean_city(df):
    df = df.copy()
    # standardize city names: spacing, title, saint and mount
    df["city_clean"] = (
        df["city"]
        .str.strip(" ,.") #remove space, commas, period
        .str.replace(r"\s+", " ", regex=True) #replace sequence whitespace with one space
        .str.title()
        .str.replace(r"\b(Of|And|Or|A)\b", lambda m: m.group().lower(), regex=True) #don't title 'a', 'of', 'and', or 'or'
        .str.replace(r"\bSt(?:\.\s*|\s+)", "Saint ", regex=True)
        .str.replace(r"\bMt(?:\.\s*|\s+)", "Mount ", regex=True)
        .str.replace(r"\bO'?\s*Fallon\b", "O'Fallon", regex=True)
        .str.replace(r"\bLand O'?\s*Lakes\b", "Land O' Lakes", regex=True)
        .str.replace(r"\bHts\b\.?", "Heights", regex=True)
        .str.replace(r"\bBlf\b\.?", "Bluffs", regex=True)
        .str.replace(r"\bBch\b\.?", "Beach", regex=True)
        .str.replace(r"\bTwp\b\.?", "Township", regex=True)
        #For directional abbreviation
        .str.replace(r"^N(?:\.\s*|\s+)", "N. ", regex=True)
        .str.replace(r"^S(?:\.\s*|\s+)", "S. ", regex=True)
        .str.replace(r"^E(?:\.\s*|\s+)", "E. ", regex=True)
        .str.replace(r"^W(?:\.\s*|\s+)", "W. ", regex=True)
    )

    #correct spelling mistakes and nicknames
    city_mapping = {
        "Philadephia": "Philadelphia",
        "Tuscon": "Tucson",
        "Saintt Petersburg": "Saint Petersburg",
        "Saint Petersurg": "Saint Petersburg",
        "Conshohoeken": "Conshohocken",
        "Thonosassa": "Thonotosassa",
        "Newtown Sqaure": "Newtown Square",
        "Newton": "Newtown",
        "Metarie": "Metairie",
        "Goodletsville": "Goodlettsville",
        "Chalemette": "Chalmette",
        "Plainfiled": "Plainfield",
        "Tierre Verde": "Tierra Verde",
        "Glenoldan": "Glenolden",
        "Indianopolis": "Indianapolis",
        "Inpolis": "Indianapolis",
        "Cedarbrook": "Cedar Brook",
        "Springhill": "Spring Hill",
        "E. Norristown": "Norristown",
        "North Redngtn Beach": "North Redington Beach",
        "Redington Shore": "Redington Shores",
        "Redingtn Shor": "Redington Shores",
        "W Cherry Hill": "Cherry Hill",
        "Town N Country": "Town and Country",
        "Town & Country": "Town and Country",
        "Town 'N' Country": "Town and Country",
        "Twn N Cntry": "Town and Country",
        "Bellville": "Belleville",
        "Belle Chase": "Belle Chasse",
        "Riveridge": "River Ridge",
        "Westhampton": "Westampton", #at hamilton,
        "Hamilton": "Hamilton Township",
        "Hamiltion": "Hamilton Township",
        "Festerville": "Feasterville",
        "Feasterville Trevose": "Feasterville-Trevose",
        "Scott Afb": "Scott Air Force Base",
        "Nw Edmonton": "NW Edmonton", #the directions

        #suffix with state
        "Riverview FL": "Riverview",
        "West Chester Pa": "West Chester",
        "Lutz Fl": "Lutz",

        # nicknames and special formatting
        "Phila": "Philadelphia",
        "Philly": "Philadelphia",
        "Mccordsville": "McCordsville",
        "Mc Cordsville": "McCordsville",
        "Temple Terr": "Temple Terrace",
        "Carney'S Point": "Carneys Point",
        "Corona De Tucson": "Corona de Tucson",
        "Saint Pete": "Saint Petersburg"
    }
    df["city_clean"] = df["city_clean"].replace(city_mapping)

    fixed = (df["city"] != df["city_clean"]).sum()
    print("clean_city: fixed",fixed,"rows")

    return df

if __name__ == "__main__":
    restaurant_cleaned=clean_city(restaurants_all)
    changed = restaurant_cleaned.loc[
        restaurant_cleaned["city"] != restaurant_cleaned["city_clean"]
    ]




