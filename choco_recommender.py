import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

st.title("ChocoRecommender")
st.write("""
    ChocoRecommender is a small 01/18/2022 #TidyTuesday project (done in Python). 
    It uses the chocolate ratings dataset to allow you to find chocolates with descriptors similar to those of your ideal chocolate. 
    This is implemented using TF-IDF vectors for each chocolate description. 
    ChocoRecommender also allows you to filter your results based on chocolate characteristics.
""")

@st.cache
def load_and_clean_data():
    df = pd.read_csv("chocolate.csv")

    df["cocoa_percent"] = df["cocoa_percent"].apply(lambda x: float(str(x).split("%")[0]))
    df["country_of_bean_origin"] = df["country_of_bean_origin"].apply(lambda x: "United States of America" if str(x) == "U.S.A." else str(x))
    df["company_location"] = df["company_location"].apply(lambda x: "United States of America" if str(x) == "U.S.A." else str(x))

    return df

def filter_data(data):
    data = data.loc[data["rating"] >= rating_filter]
    data = data.loc[data["cocoa_percent"] >= cocoa_filter]
    if maker_location != "Anywhere":
        data = data.loc[data["company_location"] == maker_location]
    if bean_origin != "Anywhere":
        data = data.loc[data["country_of_bean_origin"] == bean_origin]
    return data

def make_embeddings_and_get_neighbours(data, query, stops):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2', stop_words=stops)
    tfidf_vectorizer_result = tfidf_vectorizer.fit_transform(data["most_memorable_characteristics"].append(pd.Series(query)))
    query_result = tfidf_vectorizer_result[-1]
    tfidf_vectorizer_result = tfidf_vectorizer_result[:-1]
    neigh = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
    neigh.fit(tfidf_vectorizer_result)
    res = neigh.kneighbors(query_result, 5, return_distance=True)
    result = data.iloc[res[1].tolist()[0]][["specific_bean_origin_or_bar_name", "company_manufacturer", "company_location", "country_of_bean_origin", "cocoa_percent", "most_memorable_characteristics", "rating"]]
    result["Similarity Score"] = [f"{round((1 - x)*100, 2)}%" for x in res[0].tolist()[0]]
    result["cocoa_percent"] = result["cocoa_percent"].apply(lambda x: str(round(x))+"%")
    result["rating"] = result["rating"].apply(lambda x: str(round(x, 2)))
    result.columns = ["Name", "Maker", "Maker Location", "Cocoa Bean Origin", "Cocoa Percentage", "Notes", "Rating", "Similarity Score"]
    return result

data = load_and_clean_data()
stops = stopwords.words("english") + ["cocoa"]

query = st.text_input(
    label = "Describe your favourite chocolate here (ex: fruity, berries, floral)",
    max_chars = 35
    )

rating_filter = st.slider(
    label = "Minimum Rating",
    min_value = 0.0,
    max_value = 5.0,
    step = 0.25
)

cocoa_filter = st.slider(
    label = "Minimum Cocoa Percentage",
    min_value = 0,
    max_value = 100,
    step = 5
)

maker_location = st.selectbox(
    label = "Where should the maker be located?",
    options = ["Anywhere"] + pd.unique(data.company_location).tolist()
)

bean_origin = st.selectbox(
    label = "Where should the beans come from?",
    options = ["Anywhere"] + pd.unique(data.country_of_bean_origin).tolist()
)

data = filter_data(data)

try:
    result = make_embeddings_and_get_neighbours(data, query, stops)
    st.write(result)
except ValueError as error:
    st.write("Something went wrong. It is likely there are no chocolates that correspond to all your filters. Try relaxing some filters and trying again.")