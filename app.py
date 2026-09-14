import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from dotenv import load_dotenv
from groq import Groq

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="OTT Recommendation Agent",
    page_icon="🎬",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ENV_FILE = os.path.join(BASE_DIR, ".env")

WATCHLIST_FILE = os.path.join(DATA_DIR, "watchlist.csv")
FEEDBACK_FILE = os.path.join(DATA_DIR, "user_feedback.csv")

load_dotenv(ENV_FILE)
TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = "openai/gpt-oss-20b"

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #777;
        font-size: 18px;
        margin-bottom: 25px;
    }
    .poster-placeholder {
        width: 100%;
        height: 320px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg,#111827,#374151);
        color: white;
        font-size: 48px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA / MODELS
# ============================================================
@st.cache_data
def load_data():
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    watch_history = pd.read_csv(os.path.join(DATA_DIR, "watch_history.csv"))
    search_logs = pd.read_csv(os.path.join(DATA_DIR, "search_logs.csv"))
    reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
    recommendation_logs = pd.read_csv(
        os.path.join(DATA_DIR, "recommendation_logs.csv")
    )
    return movies, users, watch_history, search_logs, reviews, recommendation_logs


@st.cache_resource
def load_models():
    return {
        "similarity": joblib.load(
            os.path.join(MODEL_DIR, "improved_movie_similarity.pkl")
        ),
        "movie_indices": joblib.load(
            os.path.join(MODEL_DIR, "improved_movie_indices.pkl")
        ),
        "svd": joblib.load(
            os.path.join(MODEL_DIR, "improved_svd_model.pkl")
        ),
        "movie_features": joblib.load(
            os.path.join(MODEL_DIR, "improved_movie_features.pkl")
        ),
        "user_behaviour": joblib.load(
            os.path.join(MODEL_DIR, "improved_user_movie_behaviour.pkl")
        ),
        "config": joblib.load(
            os.path.join(MODEL_DIR, "improved_model_config.pkl")
        ),
    }


try:
    (
        movies,
        users,
        watch_history,
        search_logs,
        reviews,
        recommendation_logs,
    ) = load_data()
    models = load_models()
    data_loaded = True
except Exception as error:
    data_loaded = False
    st.error("❌ Error loading data or models.")
    st.exception(error)

# ============================================================
# GENERAL HELPERS
# ============================================================
def safe_value(row, column, default="Unknown"):
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    value = str(value).strip()
    return default if value == "" else value


def get_imdb_score(movie):
    for column in ["imdb_rating", "imdb_score", "imdb", "IMDb", "IMDB"]:
        if column not in movie.index:
            continue
        value = pd.to_numeric(movie[column], errors="coerce")
        if pd.isna(value):
            continue
        value = float(value)
        if value <= 0:
            return None
        if value > 10:
            value /= 10
        return float(np.clip(value, 0, 10))
    return None


def format_rating(value):
    try:
        value = float(value)
        return "N/A" if value <= 0 else f"{value:.1f}/10"
    except Exception:
        return "N/A"


def get_user_watch_history(user_id):
    if watch_history.empty or "user_id" not in watch_history.columns:
        return pd.DataFrame()
    return watch_history[
        watch_history["user_id"].astype(str) == str(user_id)
    ].copy()


def get_user_search_history(user_id):
    if search_logs.empty or "user_id" not in search_logs.columns:
        return pd.DataFrame()
    return search_logs[
        search_logs["user_id"].astype(str) == str(user_id)
    ].copy()

# ============================================================
# WATCHLIST
# ============================================================
@st.cache_data(show_spinner=False)
def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(WATCHLIST_FILE)
    except Exception:
        return pd.DataFrame()


def save_watchlist(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(WATCHLIST_FILE, index=False)
    load_watchlist.clear()


def is_movie_in_watchlist(user_id, movie_id):
    df = load_watchlist()
    if df.empty:
        return False
    if "user_id" not in df.columns or "movie_id" not in df.columns:
        return False
    return bool(
        (
            (df["user_id"].astype(str) == str(user_id))
            & (df["movie_id"].astype(str) == str(movie_id))
        ).any()
    )


def add_movie_to_watchlist(user_id, movie):
    movie_id = safe_value(movie, "movie_id", "")
    if not movie_id:
        return False
    if is_movie_in_watchlist(user_id, movie_id):
        return False

    df = load_watchlist()
    row = movie.to_dict()
    row["user_id"] = str(user_id)
    row["movie_id"] = str(movie_id)
    new_row = pd.DataFrame([row])

    updated = (
        new_row
        if df.empty
        else pd.concat([df, new_row], ignore_index=True, sort=False)
    )
    save_watchlist(updated)
    return True


def remove_movie_from_watchlist(user_id, movie_id):
    df = load_watchlist()
    if df.empty:
        return
    if "user_id" not in df.columns or "movie_id" not in df.columns:
        return

    df = df[
        ~(
            (df["user_id"].astype(str) == str(user_id))
            & (df["movie_id"].astype(str) == str(movie_id))
        )
    ].copy()
    save_watchlist(df)

# ============================================================
# FEEDBACK
# ============================================================
@st.cache_data(show_spinner=False)
def load_feedback():
    columns = ["user_id", "movie_id", "title", "feedback", "rating"]

    if not os.path.exists(FEEDBACK_FILE):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(FEEDBACK_FILE)
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        return df[columns]
    except Exception:
        return pd.DataFrame(columns=columns)


def save_feedback(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(FEEDBACK_FILE, index=False)

    # IMPORTANT FIX:
    # Only the cached recommendation function has .clear().
    load_feedback.clear()
    get_feedback_preferences.clear()
    get_user_feedback_map.clear()
    generate_recommendations_cached.clear()


@st.cache_data(show_spinner=False)
def get_user_feedback_map(user_id):
    df = load_feedback()

    if df.empty:
        return {}

    df = df[
        df["user_id"].astype(str) == str(user_id)
    ].copy()

    if df.empty:
        return {}

    df["movie_id"] = df["movie_id"].astype(str)
    df = df.drop_duplicates("movie_id", keep="last")

    return {
        str(row["movie_id"]): {
            "title": row["title"],
            "feedback": row["feedback"],
            "rating": row["rating"],
        }
        for _, row in df.iterrows()
    }


def get_movie_feedback(user_id, movie_id):
    return get_user_feedback_map(
        user_id
    ).get(str(movie_id))


def save_movie_feedback(
    user_id,
    movie_id,
    title,
    feedback_type=None,
    rating=None,
):
    df = load_feedback()

    if not df.empty:
        df = df[
            ~(
                (df["user_id"].astype(str) == str(user_id))
                & (df["movie_id"].astype(str) == str(movie_id))
            )
        ].copy()

    new_row = pd.DataFrame(
        [{
            "user_id": str(user_id),
            "movie_id": str(movie_id),
            "title": str(title),
            "feedback": feedback_type or "",
            "rating": rating if rating is not None else "",
        }]
    )

    save_feedback(
        pd.concat(
            [df, new_row],
            ignore_index=True,
        )
    )


def display_feedback_controls(user_id, movie, prefix):
    movie_id = safe_value(movie, "movie_id", "")
    title = safe_value(movie, "title", "Unknown")

    if not movie_id:
        return

    existing = get_movie_feedback(user_id, movie_id)

    current_feedback = ""
    current_rating = 0

    if existing:
        current_feedback = str(
            existing.get("feedback", "")
        ).strip()

        try:
            current_rating = int(
                float(existing.get("rating", 0))
            )
        except Exception:
            current_rating = 0

    c1, c2 = st.columns(2)

    with c1:
        if st.button(
            "👍 Like",
            key=f"{prefix}_like_{user_id}_{movie_id}",
            use_container_width=True,
        ):
            save_movie_feedback(
                user_id,
                movie_id,
                title,
                "Like",
                current_rating if current_rating > 0 else None,
            )
            st.rerun()

    with c2:
        if st.button(
            "👎 Dislike",
            key=f"{prefix}_dislike_{user_id}_{movie_id}",
            use_container_width=True,
        ):
            save_movie_feedback(
                user_id,
                movie_id,
                title,
                "Dislike",
                current_rating if current_rating > 0 else None,
            )
            st.rerun()

    rating = st.select_slider(
        "⭐ Your Rating",
        options=[1, 2, 3, 4, 5],
        value=current_rating if 1 <= current_rating <= 5 else 5,
        key=f"{prefix}_rating_{user_id}_{movie_id}",
    )

    if st.button(
        "⭐ Save Rating",
        key=f"{prefix}_save_{user_id}_{movie_id}",
        use_container_width=True,
    ):
        save_movie_feedback(
            user_id,
            movie_id,
            title,
            current_feedback or None,
            rating,
        )
        st.rerun()

# ============================================================
# LOCAL SEARCH
# ============================================================
def search_local_movies(query):
    query = str(query).strip().lower()
    if not query:
        return pd.DataFrame()

    data = movies.copy()

    fields = [
        "movie_id", "title", "genre_primary", "genre_secondary",
        "content_type", "language", "country_of_origin", "release_year"
    ]

    for field in fields:
        if field not in data.columns:
            data[field] = ""
        data[field] = data[field].fillna("").astype(str)

    data["_title"] = data["title"].str.lower().str.strip()
    data["_genre"] = data["genre_primary"].str.lower()
    data["_secondary"] = data["genre_secondary"].str.lower()
    data["_type"] = data["content_type"].str.lower()
    data["_language"] = data["language"].str.lower()
    data["_country"] = data["country_of_origin"].str.lower()
    data["_year"] = data["release_year"].str.lower()

    query_words = set(query.split())
    scores = []

    for _, row in data.iterrows():
        title = row["_title"]
        score = 0

        if title == query:
            score += 1000
        elif title.startswith(query):
            score += 800
        elif query in title:
            score += 600

        score += len(set(title.split()) & query_words) * 100

        for field, weight in [
            ("_genre", 300), ("_secondary", 250), ("_type", 200),
            ("_language", 180), ("_country", 120), ("_year", 100)
        ]:
            if query in row[field]:
                score += weight

        scores.append(score)

    data["search_relevance"] = scores
    data = data[data["search_relevance"] > 0].copy()

    if data.empty:
        return data

    data["source"] = "Local"
    data["tmdb_id"] = ""

    if "poster_url" not in data.columns:
        data["poster_url"] = ""

    if "overview" not in data.columns:
        data["overview"] = ""

    return (
        data
        .sort_values("search_relevance", ascending=False)
        .drop_duplicates("movie_id")
        .reset_index(drop=True)
        .drop(
            columns=[
                "_title", "_genre", "_secondary", "_type",
                "_language", "_country", "_year"
            ],
            errors="ignore",
        )
    )


# ============================================================
# COMBINED SEARCH
# ============================================================
def combined_search(query):
    local_results = search_local_movies(query)
    tmdb_results = search_tmdb(query)

    tmdb_error = ""

    if (
        not tmdb_results.empty
        and "_tmdb_error" in tmdb_results.columns
    ):
        tmdb_error = str(
            tmdb_results.iloc[0]["_tmdb_error"]
        )
        tmdb_results = pd.DataFrame()

    st.session_state["tmdb_error"] = tmdb_error

    if local_results.empty and tmdb_results.empty:
        return pd.DataFrame()

    if local_results.empty:
        combined = tmdb_results.copy()
    elif tmdb_results.empty:
        combined = local_results.copy()
    else:
        combined = pd.concat(
            [local_results, tmdb_results],
            ignore_index=True,
            sort=False,
        )

    required = [
        "movie_id", "title", "genre_primary", "genre_secondary",
        "content_type", "language", "country_of_origin", "release_year",
        "imdb_rating", "poster_url", "source", "search_relevance", "overview"
    ]

    for column in required:
        if column not in combined.columns:
            combined[column] = ""

    combined["search_relevance"] = pd.to_numeric(
        combined["search_relevance"],
        errors="coerce",
    ).fillna(0)

    return (
        combined
        .sort_values("search_relevance", ascending=False)
        .reset_index(drop=True)
    )

# ============================================================
# TMDB
# ============================================================
@st.cache_data(show_spinner=False)
def search_tmdb(query):
    if not TMDB_API_TOKEN:
        return pd.DataFrame()

    query = str(query).strip()
    if not query:
        return pd.DataFrame()

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            headers={
                "Authorization": f"Bearer {TMDB_API_TOKEN}",
                "accept": "application/json",
            },
            params={
                "query": query,
                "include_adult": "false",
                "language": "en-US",
                "page": 1,
            },
            timeout=8,
        )

        if response.status_code == 401:
            return pd.DataFrame(
                [{"_tmdb_error": "TMDB authentication failed"}]
            )

        if response.status_code == 429:
            return pd.DataFrame(
                [{"_tmdb_error": "TMDB rate limit reached"}]
            )

        if response.status_code != 200:
            return pd.DataFrame(
                [{
                    "_tmdb_error":
                        f"TMDB request failed: {response.status_code}"
                }]
            )

        results = response.json().get("results", [])
        if not results:
            return pd.DataFrame()

        query_norm = (
            query.lower()
            .replace("-", "")
            .replace(" ", "")
        )

        rows = []

        for item in results:
            title = str(item.get("title", "")).strip()

            title_norm = (
                title.lower()
                .replace("-", "")
                .replace(" ", "")
            )

            if title_norm == query_norm:
                relevance = 1000
            elif title_norm.startswith(query_norm):
                relevance = 800
            elif query_norm in title_norm:
                relevance = 600
            else:
                relevance = 300

            release_date = str(
                item.get("release_date", "")
            )

            release_year = (
                release_date[:4]
                if len(release_date) >= 4
                and release_date[:4].isdigit()
                else ""
            )

            rating = pd.to_numeric(
                item.get("vote_average", 0),
                errors="coerce",
            )

            rating = 0.0 if pd.isna(rating) else float(rating)

            poster_path = item.get("poster_path")
            poster_url = (
                "https://image.tmdb.org/t/p/w500" + str(poster_path)
                if poster_path
                else ""
            )

            rows.append(
                {
                    "movie_id": f"tmdb_{item.get('id', '')}",
                    "title": title or "Unknown",
                    "genre_primary": "TMDB",
                    "genre_secondary": "",
                    "content_type": "Movie",
                    "language": str(
                        item.get("original_language", "Unknown")
                        or "Unknown"
                    ).upper(),
                    "country_of_origin": "TMDB",
                    "release_year": release_year,
                    "imdb_rating": rating if rating > 0 else np.nan,
                    "poster_url": poster_url,
                    "overview": item.get("overview", ""),
                    "source": "TMDB",
                    "tmdb_id": item.get("id"),
                    "popularity": float(
                        item.get("popularity", 0) or 0
                    ),
                    "search_relevance": relevance,
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                ["search_relevance", "popularity"],
                ascending=[False, False],
            )
            .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_tmdb_movie_details(tmdb_id):
    if not TMDB_API_TOKEN or not tmdb_id:
        return {}

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            headers={
                "Authorization": f"Bearer {TMDB_API_TOKEN}",
                "accept": "application/json",
            },
            params={
                "language": "en-US",
                "append_to_response": "external_ids",
            },
            timeout=8,
        )

        if response.status_code != 200:
            return {}

        data = response.json()

        rating = pd.to_numeric(
            data.get("vote_average", 0),
            errors="coerce",
        )
        rating = 0.0 if pd.isna(rating) else float(rating)

        runtime = data.get("runtime")
        runtime_text = "N/A"

        if runtime:
            hours, minutes = divmod(int(runtime), 60)
            runtime_text = (
                f"{hours}h {minutes}m"
                if hours
                else f"{minutes} min"
            )

        external_ids = data.get(
            "external_ids",
            {}
        ) or {}

        return {
            "title": data.get("title", "Unknown"),
            "genres": [
                x.get("name")
                for x in data.get("genres", [])
                if x.get("name")
            ],
            "countries": [
                x.get("name")
                for x in data.get("production_countries", [])
                if x.get("name")
            ],
            "language": str(
                data.get("original_language", "Unknown")
                or "Unknown"
            ).upper(),
            "release_date": str(
                data.get("release_date", "")
            ),
            "tmdb_rating": rating,
            "runtime": runtime_text,
            "imdb_id": external_ids.get("imdb_id", ""),
            "overview": data.get("overview", ""),
            "tagline": data.get("tagline", ""),
        }

    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def get_tmdb_poster(title, release_year=None):
    results = search_tmdb(title)

    if results.empty:
        return ""

    target_year = None

    try:
        if release_year not in [
            None,
            "",
            "Unknown",
            "nan",
        ]:
            target_year = int(float(release_year))
    except Exception:
        target_year = None

    if target_year is not None:
        for _, row in results.iterrows():
            year = str(row.get("release_year", ""))
            if year.isdigit() and int(year) == target_year:
                return str(row.get("poster_url", ""))

    return str(results.iloc[0].get("poster_url", ""))


def get_movie_poster(movie):
    for column in [
        "poster_url",
        "poster",
        "image_url",
        "thumbnail_url",
    ]:
        if column not in movie.index:
            continue

        value = movie[column]
        if pd.isna(value):
            continue

        value = str(value).strip()
        if value.startswith(("http://", "https://")):
            return value

    return get_tmdb_poster(
        safe_value(movie, "title", ""),
        safe_value(movie, "release_year", None),
    )

# ============================================================
# FEEDBACK PREFERENCES
# ============================================================
@st.cache_data(show_spinner=False)
def get_feedback_preferences(user_id):
    feedback = load_feedback()

    result = {
        "liked_genres": [],
        "disliked_genres": [],
        "liked_types": [],
        "disliked_types": [],
        "average_rating": None,
        "feedback_count": 0,
        "likes": 0,
        "dislikes": 0,
    }

    if feedback.empty:
        return result

    feedback = feedback[
        feedback["user_id"].astype(str) == str(user_id)
    ].copy()

    if feedback.empty:
        return result

    result["feedback_count"] = len(feedback)

    result["likes"] = int(
        (
            feedback["feedback"]
            .astype(str)
            .str.lower()
            == "like"
        ).sum()
    )

    result["dislikes"] = int(
        (
            feedback["feedback"]
            .astype(str)
            .str.lower()
            == "dislike"
        ).sum()
    )

    ratings = pd.to_numeric(
        feedback["rating"],
        errors="coerce"
    ).dropna()

    if not ratings.empty:
        result["average_rating"] = float(
            ratings.mean()
        )

    if movies.empty:
        return result

    lookup = movies.copy()
    lookup["movie_id"] = lookup["movie_id"].astype(str)
    feedback["movie_id"] = feedback["movie_id"].astype(str)

    merged = feedback.merge(
        lookup[
            [
                "movie_id",
                "genre_primary",
                "content_type",
            ]
        ],
        on="movie_id",
        how="left",
    )

    for _, row in merged.iterrows():

        feedback_type = str(
            row.get("feedback", "")
        ).lower().strip()

        try:
            rating = float(
                row.get("rating", 0)
            )
        except Exception:
            rating = 0

        positive = (
            feedback_type == "like"
            or
            rating >= 4
        )

        negative = (
            feedback_type == "dislike"
            or
            (0 < rating <= 2)
        )

        genre = safe_value(
            row,
            "genre_primary",
            ""
        )

        content_type = safe_value(
            row,
            "content_type",
            ""
        )

        if positive:
            if genre:
                result["liked_genres"].append(genre)
            if content_type:
                result["liked_types"].append(content_type)

        if negative:
            if genre:
                result["disliked_genres"].append(genre)
            if content_type:
                result["disliked_types"].append(content_type)

    return result

# ============================================================
# RECOMMENDATION ENGINE
# ============================================================
@st.cache_data(
    show_spinner=False,
    max_entries=50,
)
def generate_recommendations_cached(
    user_id,
    top_n,
    feedback_signature,
):
    candidates = (
        movies
        .copy()
        .drop_duplicates("movie_id")
        .reset_index(drop=True)
    )

    candidates["movie_id"] = (
        candidates["movie_id"].astype(str)
    )

    history = get_user_watch_history(user_id)

    if history.empty:
        watched_ids = set()
    else:
        history["movie_id"] = (
            history["movie_id"].astype(str)
        )
        watched_ids = set(
            history["movie_id"]
        )

    candidates = candidates[
        ~candidates["movie_id"].isin(watched_ids)
    ].copy()

    # Respect the user's explicit "Avoid" preferences.
    # These preferences come from negative feedback and should not
    # merely receive a small score penalty.
    feedback_preferences = get_feedback_preferences(user_id)

    disliked_genres = {
        str(x).strip()
        for x in feedback_preferences.get("disliked_genres", [])
        if str(x).strip()
    }

    disliked_types = {
        str(x).strip()
        for x in feedback_preferences.get("disliked_types", [])
        if str(x).strip()
    }

    if "genre_primary" in candidates.columns and disliked_genres:
        candidates = candidates[
            ~candidates["genre_primary"].astype(str).isin(disliked_genres)
        ].copy()

    if "content_type" in candidates.columns and disliked_types:
        candidates = candidates[
            ~candidates["content_type"].astype(str).isin(disliked_types)
        ].copy()

    if candidates.empty:
        return pd.DataFrame()

    preferred_genres = []
    preferred_types = []

    if not history.empty:

        if "genre_primary" in history.columns:
            preferred_genres = (
                history["genre_primary"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

        if "content_type" in history.columns:
            preferred_types = (
                history["content_type"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

    feedback_preferences = (
        get_feedback_preferences(user_id)
    )

    liked_genres = set(
        feedback_preferences["liked_genres"]
    )

    disliked_genres = set(
        feedback_preferences["disliked_genres"]
    )

    liked_types = set(
        feedback_preferences["liked_types"]
    )

    disliked_types = set(
        feedback_preferences["disliked_types"]
    )

    searches = get_user_search_history(user_id)
    search_words = set()

    if (
        not searches.empty
        and
        "search_query" in searches.columns
    ):
        search_words = set(
            " ".join(
                searches["search_query"]
                .dropna()
                .astype(str)
                .tolist()
            )
            .lower()
            .split()
        )

    saved_indices = models["movie_indices"]

    if isinstance(saved_indices, pd.Series):
        index_dict = {
            str(k): int(v)
            for k, v in saved_indices.items()
        }
    elif isinstance(saved_indices, dict):
        index_dict = {
            str(k): int(v)
            for k, v in saved_indices.items()
        }
    else:
        index_dict = {}

    similarity = models["similarity"]
    svd = models["svd"]

    watched_indices = [
        index_dict[movie_id]
        for movie_id in watched_ids
        if movie_id in index_dict
    ]

    results = []

    for _, movie in candidates.iterrows():

        movie_id = str(movie["movie_id"])
        title = safe_value(movie, "title")
        genre = safe_value(movie, "genre_primary")
        content_type = safe_value(movie, "content_type")
        language = safe_value(movie, "language")

        # Content
        content_score = 0.0
        candidate_index = index_dict.get(movie_id)

        if (
            candidate_index is not None
            and
            watched_indices
        ):
            maximum = 0.0

            for watched_index in watched_indices:
                try:
                    value = float(
                        similarity[
                            watched_index,
                            candidate_index
                        ]
                    )
                    maximum = max(
                        maximum,
                        value
                    )
                except Exception:
                    continue

            content_score = maximum

        # SVD
        svd_score = 0.0

        try:
            prediction = svd.predict(
                str(user_id),
                movie_id
            )

            svd_score = float(
                np.clip(
                    float(prediction.est) / 5.0,
                    0,
                    1,
                )
            )
        except Exception:
            pass

        # Behaviour from history
        genre_history_score = (
            1.0
            if genre in preferred_genres
            else 0.0
        )

        type_history_score = (
            1.0
            if content_type in preferred_types
            else 0.0
        )

        history_behaviour_score = (
            0.7 * genre_history_score
            +
            0.3 * type_history_score
        )

        # Explicit feedback
        feedback_genre_score = 0.0

        if genre in liked_genres:
            feedback_genre_score += 1.0

        if genre in disliked_genres:
            feedback_genre_score -= 1.0

        feedback_type_score = 0.0

        if content_type in liked_types:
            feedback_type_score += 1.0

        if content_type in disliked_types:
            feedback_type_score -= 1.0

        explicit_feedback_score = (
            0.7 * feedback_genre_score
            +
            0.3 * feedback_type_score
        )

        explicit_feedback_score = float(
            np.clip(
                explicit_feedback_score,
                -1,
                1,
            )
        )

        feedback_positive_score = (
            explicit_feedback_score + 1
        ) / 2

        behaviour_score = (
            0.60 * history_behaviour_score
            +
            0.40 * feedback_positive_score
        )

        behaviour_score = float(
            np.clip(
                behaviour_score,
                0,
                1,
            )
        )

        # Search
        search_score = 0.0

        if search_words:

            movie_words = set(
                (
                    f"{title} "
                    f"{genre} "
                    f"{content_type} "
                    f"{language}"
                )
                .lower()
                .split()
            )

            search_score = (
                len(
                    search_words
                    &
                    movie_words
                )
                /
                len(search_words)
            )

        # Quality
        imdb_score = get_imdb_score(movie)

        quality_score = (
            imdb_score / 10.0
            if imdb_score is not None
            else 0.0
        )

        # Dislike penalty
        dislike_penalty = 0.0

        if genre in disliked_genres:
            dislike_penalty += 0.20

        if content_type in disliked_types:
            dislike_penalty += 0.10

        # Hybrid
        hybrid_score = (
            0.30 * content_score
            +
            0.25 * svd_score
            +
            0.20 * behaviour_score
            +
            0.10 * search_score
            +
            0.15 * quality_score
            -
            dislike_penalty
        )

        hybrid_score = float(
            np.clip(
                hybrid_score,
                0,
                1,
            )
        )

        signals = []

        if content_score > 0.15:
            signals.append(
                "similar to your watched content"
            )

        if svd_score > 0.50:
            signals.append(
                "matches collaborative preferences"
            )

        if history_behaviour_score > 0.50:
            signals.append(
                "matches your viewing behaviour"
            )

        if genre in liked_genres:
            signals.append(
                "matches genres you liked"
            )

        if content_type in liked_types:
            signals.append(
                "matches content types you liked"
            )

        if search_score > 0:
            signals.append(
                "matches your search interests"
            )

        if quality_score >= 0.70:
            signals.append(
                "has a strong movie rating"
            )

        if genre in disliked_genres:
            signals.append(
                "reduced because of a disliked genre"
            )

        if content_type in disliked_types:
            signals.append(
                "reduced because of a disliked content type"
            )

        if not signals:
            signals.append(
                "matches your overall preferences"
            )

        results.append(
            {
                "movie_id": movie_id,
                "title": title,
                "genre": genre,
                "content_type": content_type,
                "language": language,
                "release_year": safe_value(
                    movie,
                    "release_year",
                    ""
                ),
                "imdb_score": imdb_score,
                "content_score": content_score,
                "svd_score": svd_score,
                "behaviour_score": behaviour_score,
                "search_score": search_score,
                "quality_score": quality_score,
                "hybrid_score": hybrid_score,
                "signals": ", ".join(signals),
                "poster_url": "",
            }
        )

    results.sort(
        key=lambda x:
            x["hybrid_score"],
        reverse=True,
    )

    selected = []
    genre_count = {}

    for item in results:

        genre = item["genre"]

        if genre_count.get(genre, 0) >= 3:
            continue

        selected.append(item)
        genre_count[genre] = (
            genre_count.get(genre, 0)
            +
            1
        )

        if len(selected) >= top_n:
            break

    final_df = pd.DataFrame(selected)

    if not final_df.empty:

        lookup = candidates.set_index(
            "movie_id"
        )

        posters = []

        for movie_id in final_df["movie_id"]:

            try:

                original = lookup.loc[
                    str(movie_id)
                ]

                posters.append(
                    get_movie_poster(
                        original
                    )
                )

            except Exception:

                posters.append("")

        final_df["poster_url"] = posters

    return final_df


def generate_recommendations(
    user_id,
    top_n
):
    feedback = load_feedback()

    if feedback.empty:
        feedback_signature = "empty"
    else:
        user_feedback = feedback[
            feedback[
                "user_id"
            ].astype(str)
            ==
            str(user_id)
        ].copy()

        if user_feedback.empty:
            feedback_signature = "none"
        else:
            feedback_signature = (
                user_feedback[
                    [
                        "movie_id",
                        "feedback",
                        "rating",
                    ]
                ]
                .astype(str)
                .to_csv(index=False)
            )

    return generate_recommendations_cached(
        str(user_id),
        int(top_n),
        feedback_signature,
    )

# ============================================================
# GENAI - GROQ
# ============================================================
def generate_genai_explanation(api_key, movie):
    """Generate a short recommendation explanation using Groq.

    Uses the current Groq GPT-OSS 20B model and the REST API directly.
    This avoids SDK/model-parameter mismatches and explicitly disables
    reasoning output because this feature only needs one short sentence.
    """

    if not api_key:
        return {
            "status": "error",
            "text": "Groq API key was not found in the .env file.",
        }

    title = str(movie.get("title", "Unknown"))
    genre = str(movie.get("genre", "Unknown"))
    content_type = str(movie.get("content_type", "Unknown"))
    language = str(movie.get("language", "Unknown"))
    rating = format_rating(movie.get("imdb_score"))
    signals = str(
        movie.get(
            "signals",
            "matches your overall preferences",
        )
    )

    prompt = f"""
You are an OTT recommendation assistant.

Movie: {title}
Genre: {genre}
Type: {content_type}
Language: {language}
IMDb rating: {rating}
Recommendation signals: {signals}

Write exactly ONE short natural sentence explaining why this movie was recommended.

Rules:
- Use only the information above.
- Do not invent user activity.
- Do not claim the user watched, liked, searched for, or rated this movie.
- Do not mention technical models, algorithms, scores, Groq, or AI.
- Return only the final sentence.
"""

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 0.2,
                "max_completion_tokens": 1024,
                "reasoning_effort": "low",
                "include_reasoning": False,
                "stream": False,
            },
            timeout=30,
        )

        if response.status_code != 200:
            try:
                error_data = response.json()
                error_message = (
                    error_data
                    .get("error", {})
                    .get("message", "")
                )
            except Exception:
                error_message = response.text

            error_message = str(error_message).strip()
            if not error_message:
                error_message = "Unknown Groq API error."

            print("\n" + "=" * 70)
            print("GROQ API ERROR")
            print("HTTP STATUS:", response.status_code)
            print("MESSAGE:", error_message)
            print("=" * 70 + "\n")

            if response.status_code == 401:
                return {
                    "status": "error",
                    "text": "Groq API key is invalid or expired.",
                }

            if response.status_code == 403:
                return {
                    "status": "error",
                    "text": "Groq API access was denied.",
                }

            if response.status_code == 404:
                return {
                    "status": "error",
                    "text": f"Groq model '{GROQ_MODEL}' was not found or is unavailable.",
                }

            if response.status_code == 429:
                return {
                    "status": "limit",
                    "text": "Groq rate limit reached. Please try again shortly.",
                }

            return {
                "status": "error",
                "text": f"Groq API error {response.status_code}: {error_message}",
            }

        data = response.json()
        choices = data.get("choices", []) or []

        if not choices:
            return {
                "status": "error",
                "text": "Groq returned no explanation.",
            }

        message = choices[0].get("message", {}) or {}
        content = message.get("content")

        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                if isinstance(part, dict)
                else str(part)
                for part in content
            )

        if content:
            text = str(content).strip()
            if text:
                text = text.replace("<think>", "").replace("</think>", "")
                text = text.strip().strip('"').strip()
                return {
                    "status": "success",
                    "text": text,
                }

        # Some GPT-OSS responses may expose the answer in a different field.
        for key in ["answer", "text", "output"]:
            value = message.get(key)
            if value:
                text = str(value).strip().strip('"').strip()
                if text:
                    return {
                        "status": "success",
                        "text": text,
                    }

        reasoning = message.get("reasoning")
        print("Groq returned no content. Reasoning field:", reasoning)
        return {
            "status": "error",
            "text": "Groq generated no final answer. Please try clicking again.",
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "text": "Groq request timed out. Please try again.",
        }

    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "text": "Could not connect to Groq. Check your internet connection.",
        }

    except Exception as error:
        error_text = str(error).strip()

        print("\n" + "=" * 70)
        print("GROQ PYTHON ERROR")
        print(error_text)
        print("=" * 70 + "\n")

        return {
            "status": "error",
            "text": f"Groq error: {error_text}",
        }


# ============================================================
# SIDEBAR
# ============================================================
selected_user = None
number_of_recommendations = 10
groq_api_key = GROQ_API_KEY

st.sidebar.title(
    "🎬 Recommendation Settings"
)

if data_loaded:

    users["user_id"] = users[
        "user_id"
    ].astype(str)

    user_ids = sorted(
        users["user_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if user_ids:

        selected_user = st.sidebar.selectbox(
            "Select User",
            user_ids,
        )

    number_of_recommendations = st.sidebar.slider(
        "Number of Recommendations",
        5,
        10,
        10,
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "🤖 GenAI Settings"
    )

    if GROQ_API_KEY:
        st.sidebar.success("✅ Groq connected")
        st.sidebar.caption(f"Model: {GROQ_MODEL}")
    else:
        st.sidebar.error("❌ GROQ_API_KEY not found in .env")

    st.sidebar.markdown("---")

    if TMDB_API_TOKEN:
        st.sidebar.success(
            "✅ TMDB connected"
        )
    else:
        st.sidebar.error(
            "❌ TMDB token not found"
        )

    watchlist = load_watchlist()
    feedback = load_feedback()

    watchlist_count = 0
    feedback_count = 0

    if (
        not watchlist.empty
        and
        "user_id" in watchlist.columns
    ):
        watchlist_count = int(
            (
                watchlist["user_id"].astype(str)
                ==
                str(selected_user)
            ).sum()
        )

    if (
        not feedback.empty
        and
        "user_id" in feedback.columns
    ):
        feedback_count = int(
            (
                feedback["user_id"].astype(str)
                ==
                str(selected_user)
            ).sum()
        )

    st.sidebar.caption(
        f"❤️ Watchlist: {watchlist_count}"
    )

    st.sidebar.caption(
        f"⭐ Feedback: {feedback_count}"
    )

# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🎬 OTT Recommendation Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Personalized movie recommendations using '
    'Content + SVD + Behaviour + Search + Quality'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# SEARCH
# ============================================================
if data_loaded:

    st.markdown("---")
    st.subheader("🔎 Search Movies")

    st.caption(
        "Search your local dataset and TMDB. "
        "Example: Spider-Man, Avengers, Batman, Inception."
    )

    search_query = st.text_input(
        "Search",
        placeholder=(
            "Spider-Man, Avengers, Batman, Inception..."
        ),
        key="movie_search_box",
    )

    if search_query.strip():

        with st.spinner(
            "Searching local movies and TMDB..."
        ):

            search_results = combined_search(
                search_query
            )

        tmdb_error = st.session_state.get(
            "tmdb_error",
            ""
        )

        if tmdb_error:
            st.warning(
                f"⚠️ {tmdb_error}"
            )

        if search_results.empty:

            st.warning(
                f'No movies found for "{search_query}".'
            )

        else:

            local_count = int(
                (
                    search_results[
                        "source"
                    ]
                    ==
                    "Local"
                ).sum()
            )

            tmdb_count = int(
                (
                    search_results[
                        "source"
                    ]
                    ==
                    "TMDB"
                ).sum()
            )

            st.success(
                f'Found {len(search_results)} '
                f'results for "{search_query}". '
                f'Local: {local_count} | '
                f'TMDB: {tmdb_count}'
            )

            st.markdown(
                "### 🎯 Filters"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                selected_source = st.selectbox(
                    "Source",
                    [
                        "All",
                        "Local",
                        "TMDB",
                    ],
                    key="search_source_filter",
                )

            with c2:
                genres = sorted(
                    search_results[
                        "genre_primary"
                    ]
                    .fillna("Unknown")
                    .astype(str)
                    .replace("", "Unknown")
                    .unique()
                    .tolist()
                )

                selected_genre = st.selectbox(
                    "Genre",
                    ["All"] + genres,
                    key="search_genre_filter",
                )

            with c3:
                languages = sorted(
                    search_results[
                        "language"
                    ]
                    .fillna("Unknown")
                    .astype(str)
                    .replace("", "Unknown")
                    .unique()
                    .tolist()
                )

                selected_language = st.selectbox(
                    "Language",
                    ["All"] + languages,
                    key="search_language_filter",
                )

            c4, c5, c6 = st.columns(3)

            with c4:

                types = sorted(
                    search_results[
                        "content_type"
                    ]
                    .fillna("Unknown")
                    .astype(str)
                    .replace("", "Unknown")
                    .unique()
                    .tolist()
                )

                selected_type = st.selectbox(
                    "Content Type",
                    ["All"] + types,
                    key="search_type_filter",
                )

            with c5:

                search_results["_imdb"] = (
                    search_results.apply(
                        get_imdb_score,
                        axis=1
                    )
                )

                minimum_imdb = st.slider(
                    "Minimum IMDb Rating",
                    0.0,
                    10.0,
                    0.0,
                    0.5,
                    key="search_imdb_filter",
                )

            with c6:

                sort_option = st.selectbox(
                    "Sort By",
                    [
                        "Relevance",
                        "IMDb Rating",
                        "Release Year",
                        "Title",
                    ],
                    key="search_sort_filter",
                )

            filtered = search_results.copy()

            if selected_source != "All":
                filtered = filtered[
                    filtered["source"]
                    ==
                    selected_source
                ]

            if selected_genre != "All":
                filtered = filtered[
                    filtered["genre_primary"]
                    .fillna("Unknown")
                    .astype(str)
                    ==
                    selected_genre
                ]

            if selected_language != "All":
                filtered = filtered[
                    filtered["language"]
                    .fillna("Unknown")
                    .astype(str)
                    ==
                    selected_language
                ]

            if selected_type != "All":
                filtered = filtered[
                    filtered["content_type"]
                    .fillna("Unknown")
                    .astype(str)
                    ==
                    selected_type
                ]

            if minimum_imdb > 0:
                filtered = filtered[
                    filtered["_imdb"] >= minimum_imdb
                ]

            filtered["_year"] = pd.to_numeric(
                filtered["release_year"],
                errors="coerce"
            )

            if sort_option == "IMDb Rating":

                filtered = filtered.sort_values(
                    "_imdb",
                    ascending=False,
                    na_position="last"
                )

            elif sort_option == "Release Year":

                filtered = filtered.sort_values(
                    "_year",
                    ascending=False,
                    na_position="last"
                )

            elif sort_option == "Title":

                filtered = filtered.sort_values(
                    "title",
                    key=lambda x:
                        x.astype(str).str.lower()
                )

            else:

                filtered = filtered.sort_values(
                    "search_relevance",
                    ascending=False
                )

            filtered = filtered.reset_index(
                drop=True
            )

            st.write(
                f"**{len(filtered)}** "
                "movies match the current search and filters."
            )

            if filtered.empty:

                st.warning(
                    "No movies match the selected filters."
                )

            else:

                for start in range(
                    0,
                    len(filtered),
                    4
                ):

                    current = filtered.iloc[
                        start:start + 4
                    ]

                    columns = st.columns(4)

                    for column, (
                        result_index,
                        movie
                    ) in zip(
                        columns,
                        current.iterrows()
                    ):

                        with column:

                            poster_url = safe_value(
                                movie,
                                "poster_url",
                                ""
                            )

                            if not poster_url:

                                poster_url = get_movie_poster(
                                    movie
                                )

                            if poster_url:

                                st.image(
                                    poster_url,
                                    use_container_width=True
                                )

                            else:

                                st.markdown(
                                    """
                                    <div class="poster-placeholder">
                                        🎬
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            title = safe_value(
                                movie,
                                "title"
                            )

                            source = safe_value(
                                movie,
                                "source"
                            )

                            movie_id = safe_value(
                                movie,
                                "movie_id",
                                ""
                            )

                            st.markdown(
                                f"### {title}"
                            )

                            st.caption(
                                "🌐 TMDB"
                                if source == "TMDB"
                                else
                                "📁 Local Dataset"
                            )

                            st.caption(
                                f"ID: {movie_id}"
                            )

                            st.write(
                                "🎭 "
                                +
                                safe_value(
                                    movie,
                                    "genre_primary"
                                )
                            )

                            st.write(
                                "🎞️ "
                                +
                                safe_value(
                                    movie,
                                    "content_type"
                                )
                            )

                            st.write(
                                "🌐 "
                                +
                                safe_value(
                                    movie,
                                    "language"
                                )
                            )

                            year = safe_value(
                                movie,
                                "release_year",
                                ""
                            )

                            if year:
                                st.write(
                                    f"📅 {year}"
                                )

                            st.write(
                                "⭐ IMDb: "
                                +
                                format_rating(
                                    get_imdb_score(
                                        movie
                                    )
                                )
                            )

                            if is_movie_in_watchlist(
                                selected_user,
                                movie_id
                            ):

                                if st.button(
                                    "💔 Remove from Watchlist",
                                    key=(
                                        f"search_remove_"
                                        f"{selected_user}_"
                                        f"{movie_id}_"
                                        f"{result_index}"
                                    ),
                                    use_container_width=True,
                                ):

                                    remove_movie_from_watchlist(
                                        selected_user,
                                        movie_id
                                    )

                                    st.rerun()

                            else:

                                if st.button(
                                    "❤️ Add to Watchlist",
                                    key=(
                                        f"search_add_"
                                        f"{selected_user}_"
                                        f"{movie_id}_"
                                        f"{result_index}"
                                    ),
                                    use_container_width=True,
                                ):

                                    add_movie_to_watchlist(
                                        selected_user,
                                        movie
                                    )

                                    st.rerun()

                            with st.expander(
                                "⭐ Rate this movie"
                            ):

                                display_feedback_controls(
                                    selected_user,
                                    movie,
                                    f"search_{result_index}"
                                )

                            if source == "TMDB":

                                tmdb_id = movie.get(
                                    "tmdb_id"
                                )

                                if st.button(
                                    "📋 View TMDB Details",
                                    key=(
                                        f"details_"
                                        f"{selected_user}_"
                                        f"{tmdb_id}_"
                                        f"{result_index}"
                                    ),
                                    use_container_width=True,
                                ):

                                    details = (
                                        get_tmdb_movie_details(
                                            tmdb_id
                                        )
                                    )

                                    if details:

                                        genres = details.get(
                                            "genres",
                                            []
                                        )

                                        st.write(
                                            "🎭 Genres: "
                                            +
                                            (
                                                ", ".join(genres)
                                                if genres
                                                else "N/A"
                                            )
                                        )

                                        st.write(
                                            "🎞️ Type: Movie"
                                        )

                                        st.write(
                                            "🌐 Language: "
                                            +
                                            str(
                                                details.get(
                                                    "language",
                                                    "N/A"
                                                )
                                            )
                                        )

                                        st.write(
                                            "📅 Release Date: "
                                            +
                                            (
                                                details.get(
                                                    "release_date"
                                                )
                                                or "N/A"
                                            )
                                        )

                                        st.write(
                                            "⭐ TMDB Rating: "
                                            +
                                            format_rating(
                                                details.get(
                                                    "tmdb_rating"
                                                )
                                            )
                                        )

                                        st.write(
                                            "⏱️ Runtime: "
                                            +
                                            str(
                                                details.get(
                                                    "runtime",
                                                    "N/A"
                                                )
                                            )
                                        )

                                        imdb_id = details.get(
                                            "imdb_id",
                                            ""
                                        )

                                        if imdb_id:

                                            st.write(
                                                f"🎬 IMDb ID: `{imdb_id}`"
                                            )

                                            st.markdown(
                                                f"[🔗 Open IMDb]"
                                                f"(https://www.imdb.com/title/"
                                                f"{imdb_id}/)"
                                            )

                                        else:

                                            st.write(
                                                "🎬 IMDb ID: N/A"
                                            )

                                        overview = details.get(
                                            "overview",
                                            ""
                                        )

                                        if overview:

                                            with st.expander(
                                                "ℹ️ Overview"
                                            ):

                                                st.write(
                                                    overview
                                                )

                                    else:

                                        st.warning(
                                            "TMDB movie details could not be loaded."
                                        )

                            else:

                                overview = safe_value(
                                    movie,
                                    "overview",
                                    ""
                                )

                                if overview:

                                    with st.expander(
                                        "ℹ️ Overview"
                                    ):

                                        st.write(
                                            overview
                                        )

# ============================================================
# USER DASHBOARD
# ============================================================
if data_loaded and selected_user:

    st.markdown("---")

    st.subheader(
        "👤 User Information"
    )

    user_data = users[
        users["user_id"].astype(str)
        ==
        str(selected_user)
    ]

    if not user_data.empty:

        user = user_data.iloc[0]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Age",
                safe_value(user, "age")
            )

        with c2:
            st.metric(
                "Gender",
                safe_value(user, "gender")
            )

        with c3:
            st.metric(
                "Country",
                safe_value(user, "country")
            )

    user_history = (
        get_user_watch_history(
            selected_user
        )
    )

    user_searches = (
        get_user_search_history(
            selected_user
        )
    )

    if (
        not recommendation_logs.empty
        and
        "user_id" in recommendation_logs.columns
    ):

        user_recommendation_logs = (
            recommendation_logs[
                recommendation_logs[
                    "user_id"
                ].astype(str)
                ==
                str(selected_user)
            ]
        )

    else:

        user_recommendation_logs = pd.DataFrame()

    feedback = load_feedback()

    if (
        not feedback.empty
        and
        "user_id" in feedback.columns
    ):

        current_user_feedback = (
            feedback[
                feedback[
                    "user_id"
                ].astype(str)
                ==
                str(selected_user)
            ]
        )

    else:

        current_user_feedback = pd.DataFrame()

    st.subheader(
        "📊 User Activity"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Watched Movies",
            len(user_history)
        )

    with c2:
        st.metric(
            "Searches",
            len(user_searches)
        )

    with c3:
        st.metric(
            "Recommendations",
            len(user_recommendation_logs)
        )

    with c4:
        st.metric(
            "Feedback",
            len(current_user_feedback)
        )

    # ========================================================
    # PERSONALIZATION
    # ========================================================
    st.markdown("---")

    st.subheader(
        "🎯 Your Personalization Profile"
    )

    feedback_preferences = (
        get_feedback_preferences(
            selected_user
        )
    )

    history_genres = []
    history_types = []

    if not user_history.empty:

        if "genre_primary" in user_history.columns:

            history_genres = (
                user_history[
                    "genre_primary"
                ]
                .dropna()
                .astype(str)
                .value_counts()
                .head(5)
                .index
                .tolist()
            )

        if "content_type" in user_history.columns:

            history_types = (
                user_history[
                    "content_type"
                ]
                .dropna()
                .astype(str)
                .value_counts()
                .head(3)
                .index
                .tolist()
            )

    liked_genres = list(
        dict.fromkeys(
            history_genres
            +
            feedback_preferences[
                "liked_genres"
            ]
        )
    )

    liked_types = list(
        dict.fromkeys(
            history_types
            +
            feedback_preferences[
                "liked_types"
            ]
        )
    )

    disliked_items = list(
        dict.fromkeys(
            feedback_preferences[
                "disliked_genres"
            ]
            +
            feedback_preferences[
                "disliked_types"
            ]
        )
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            "#### ❤️ Favorite Genres"
        )

        if liked_genres:

            for item in liked_genres[:5]:
                st.write(
                    f"• {item}"
                )

        else:

            st.caption(
                "Not enough data yet"
            )

    with p2:

        st.markdown(
            "#### 🎞️ Favorite Types"
        )

        if liked_types:

            for item in liked_types[:5]:
                st.write(
                    f"• {item}"
                )

        else:

            st.caption(
                "Not enough data yet"
            )

    with p3:

        st.markdown(
            "#### 🚫 Avoid"
        )

        if disliked_items:

            for item in disliked_items[:5]:
                st.write(
                    f"• {item}"
                )

        else:

            st.caption(
                "No dislikes recorded"
            )

    p4, p5, p6 = st.columns(3)

    with p4:

        st.metric(
            "👍 Likes",
            feedback_preferences[
                "likes"
            ]
        )

    with p5:

        st.metric(
            "👎 Dislikes",
            feedback_preferences[
                "dislikes"
            ]
        )

    with p6:

        average = (
            feedback_preferences[
                "average_rating"
            ]
        )

        st.metric(
            "⭐ Average Rating",
            (
                "N/A"
                if average is None
                else
                f"{average:.1f}/5"
            )
        )

    if (
        feedback_preferences[
            "feedback_count"
        ]
        >
        0
    ):

        st.info(
            "🧠 Your feedback is influencing "
            "the recommendation behaviour score."
        )

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================
    st.markdown("---")

    st.subheader(
        f"🎯 Top Recommendations for {selected_user}"
    )

    st.caption(
        "Ranked using content similarity, collaborative "
        "preferences, viewing behaviour, search interests, "
        "user feedback, and movie quality."
    )

    with st.spinner(
        "Generating personalized recommendations..."
    ):

        recommendations = (
            generate_recommendations(
                selected_user,
                number_of_recommendations
            )
        )

    if recommendations.empty:

        st.warning(
            "No recommendations available."
        )

    else:

        st.success(
            f"Found {len(recommendations)} "
            "personalized recommendations."
        )

        for start in range(
            0,
            len(recommendations),
            4
        ):

            current = recommendations.iloc[
                start:start + 4
            ]

            columns = st.columns(4)

            for column, (
                dataframe_index,
                movie
            ) in zip(
                columns,
                current.iterrows()
            ):

                with column:

                    position = (
                        recommendations.index.get_loc(
                            dataframe_index
                        )
                        + 1
                    )

                    title = str(
                        movie["title"]
                    )

                    movie_id = str(
                        movie["movie_id"]
                    )

                    poster_url = str(
                        movie.get(
                            "poster_url",
                            ""
                        )
                    )

                    if poster_url.strip():

                        st.image(
                            poster_url,
                            use_container_width=True
                        )

                    else:

                        st.markdown(
                            """
                            <div class="poster-placeholder">
                                🎬
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"### #{position} {title}"
                    )

                    st.caption(
                        f"Movie ID: {movie_id}"
                    )

                    st.write(
                        f"🎭 {movie['genre']}"
                    )

                    st.write(
                        f"🎞️ {movie['content_type']}"
                    )

                    st.write(
                        f"🌐 {movie['language']}"
                    )

                    release_year = str(
                        movie.get(
                            "release_year",
                            ""
                        )
                    )

                    if release_year:
                        st.write(
                            f"📅 {release_year}"
                        )

                    st.write(
                        "⭐ IMDb: "
                        +
                        format_rating(
                            movie.get(
                                "imdb_score"
                            )
                        )
                    )

                    st.metric(
                        "Recommendation Score",
                        f"{float(movie['hybrid_score']):.3f}"
                    )

                    with st.expander(
                        "💡 Why this movie?"
                    ):

                        signal_list = [
                            x.strip()
                            for x in str(
                                movie.get(
                                    "signals",
                                    ""
                                )
                            ).split(",")
                            if x.strip()
                        ]

                        for signal in signal_list:

                            if "reduced" in signal.lower():

                                st.write(
                                    f"⚠️ {signal}"
                                )

                            else:

                                st.write(
                                    f"✅ {signal}"
                                )

                    with st.expander(
                        "⭐ Rate this recommendation"
                    ):

                        display_feedback_controls(
                            selected_user,
                            movie,
                            (
                                f"recommendation_"
                                f"{position}_"
                                f"{movie_id}"
                            )
                        )

                    with st.expander(
                        "📊 Score Details"
                    ):

                        score_table = pd.DataFrame({

                            "Signal": [
                                "Content Similarity",
                                "SVD Collaborative",
                                "User Behaviour + Feedback",
                                "Search Interest",
                                "Movie Quality",
                            ],

                            "Score": [
                                round(
                                    float(
                                        movie[
                                            "content_score"
                                        ]
                                    ),
                                    3
                                ),
                                round(
                                    float(
                                        movie[
                                            "svd_score"
                                        ]
                                    ),
                                    3
                                ),
                                round(
                                    float(
                                        movie[
                                            "behaviour_score"
                                        ]
                                    ),
                                    3
                                ),
                                round(
                                    float(
                                        movie[
                                            "search_score"
                                        ]
                                    ),
                                    3
                                ),
                                round(
                                    float(
                                        movie[
                                            "quality_score"
                                        ]
                                    ),
                                    3
                                ),
                            ],

                            "Weight": [
                                "30%",
                                "25%",
                                "20%",
                                "10%",
                                "15%",
                            ],
                        })

                        st.dataframe(
                            score_table,
                            use_container_width=True,
                            hide_index=True,
                        )

                    button_key = (
                        f"why_"
                        f"{selected_user}_"
                        f"{position}_"
                        f"{movie_id}"
                    )

                    if st.button(
                        "🤖 Why recommended?",
                        key=button_key,
                        use_container_width=True,
                    ):

                        if not groq_api_key:

                            st.warning(
                                "Please add GROQ_API_KEY to your .env file."
                            )

                        else:

                            explanation = (
                                generate_genai_explanation(
                                    groq_api_key,
                                    movie
                                )
                            )

                            if (
                                explanation["status"]
                                ==
                                "success"
                            ):

                                st.info(
                                    "🤖 "
                                    +
                                    explanation["text"]
                                )

                            else:

                                st.warning(
                                    "🤖 "
                                    +
                                    explanation["text"]
                                )

    # ========================================================
    # TABS
    # ========================================================
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = (
        st.tabs(
            [
                "🎯 Recommendations",
                "👤 Personalization",
                "👀 Watch History",
                "🔎 Search History",
                "❤️ Watchlist",
                "⭐ My Ratings",
                "🏗️ Architecture",
            ]
        )
    )

    with tab1:

        st.subheader(
            "Recommendation Components"
        )

        components = pd.DataFrame({

            "Component": [
                "Content Similarity",
                "SVD Collaborative",
                "User Behaviour + Feedback",
                "Search Interest",
                "Movie Quality",
            ],

            "Weight": [
                "30%",
                "25%",
                "20%",
                "10%",
                "15%",
            ],
        })

        st.dataframe(
            components,
            use_container_width=True,
            hide_index=True,
        )

    with tab2:

        st.subheader(
            f"🎯 Personalization Profile - {selected_user}"
        )

        st.markdown(
            "### ❤️ What you like"
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "#### Favorite Genres"
            )

            if liked_genres:

                for genre in liked_genres[:10]:

                    st.success(
                        f"❤️ {genre}"
                    )

            else:

                st.info(
                    "No strong genre preference detected yet."
                )

        with c2:

            st.markdown(
                "#### Favorite Content Types"
            )

            if liked_types:

                for content_type in liked_types[:10]:

                    st.success(
                        f"🎞️ {content_type}"
                    )

            else:

                st.info(
                    "No strong content-type preference detected yet."
                )

        st.markdown(
            "### 🚫 What you avoid"
        )

        if disliked_items:

            for item in disliked_items[:10]:

                st.warning(
                    f"🚫 {item}"
                )

        else:

            st.info(
                "No dislikes recorded yet."
            )

        st.markdown(
            "### 📊 Preference Statistics"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Watched",
                len(user_history)
            )

        with c2:
            st.metric(
                "Searches",
                len(user_searches)
            )

        with c3:
            st.metric(
                "Feedback",
                feedback_preferences[
                    "feedback_count"
                ]
            )

        with c4:

            avg = (
                feedback_preferences[
                    "average_rating"
                ]
            )

            st.metric(
                "Avg Rating",
                (
                    "N/A"
                    if avg is None
                    else
                    f"{avg:.1f}/5"
                )
            )

        st.markdown(
            "### 🧠 How personalization works"
        )

        st.write(
            "The system combines watch history, search history, "
            "likes, dislikes, ratings, collaborative preferences, "
            "content similarity, and movie quality."
        )

    with tab3:

        st.subheader(
            f"👀 Watch History - {selected_user}"
        )

        if user_history.empty:

            st.info(
                "No watch history available."
            )

        else:

            st.dataframe(
                user_history,
                use_container_width=True,
                hide_index=True,
            )

    with tab4:

        st.subheader(
            f"🔎 Search History - {selected_user}"
        )

        if user_searches.empty:

            st.info(
                "No search history available."
            )

        else:

            st.dataframe(
                user_searches,
                use_container_width=True,
                hide_index=True,
            )

    with tab5:

        st.subheader(
            f"❤️ Watchlist - {selected_user}"
        )

        watchlist = load_watchlist()

        if watchlist.empty:

            st.info(
                "Your watchlist is empty."
            )

        elif (
            "user_id" not in watchlist.columns
            or
            "movie_id" not in watchlist.columns
        ):

            st.info(
                "Your watchlist is empty."
            )

        else:

            user_watchlist = watchlist[
                watchlist[
                    "user_id"
                ].astype(str)
                ==
                str(selected_user)
            ].copy()

            if user_watchlist.empty:

                st.info(
                    "Your watchlist is empty."
                )

            else:

                st.success(
                    f"{len(user_watchlist)} movie(s) "
                    "in your watchlist."
                )

                for start in range(
                    0,
                    len(user_watchlist),
                    4
                ):

                    current = user_watchlist.iloc[
                        start:start + 4
                    ]

                    columns = st.columns(4)

                    for column, (
                        watch_index,
                        movie
                    ) in zip(
                        columns,
                        current.iterrows()
                    ):

                        with column:

                            title = safe_value(
                                movie,
                                "title",
                                "Unknown"
                            )

                            movie_id = safe_value(
                                movie,
                                "movie_id",
                                ""
                            )

                            poster_url = safe_value(
                                movie,
                                "poster_url",
                                ""
                            )

                            if poster_url:

                                st.image(
                                    poster_url,
                                    use_container_width=True
                                )

                            else:

                                st.markdown(
                                    """
                                    <div class="poster-placeholder">
                                        🎬
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                            st.markdown(
                                f"### 🎬 {title}"
                            )

                            st.caption(
                                f"ID: {movie_id}"
                            )

                            st.write(
                                "🎭 "
                                +
                                safe_value(
                                    movie,
                                    "genre_primary"
                                )
                            )

                            st.write(
                                "🎞️ "
                                +
                                safe_value(
                                    movie,
                                    "content_type"
                                )
                            )

                            st.write(
                                "🌐 "
                                +
                                safe_value(
                                    movie,
                                    "language"
                                )
                            )

                            st.write(
                                "⭐ IMDb: "
                                +
                                format_rating(
                                    get_imdb_score(
                                        movie
                                    )
                                )
                            )

                            with st.expander(
                                "⭐ Rate this movie"
                            ):

                                display_feedback_controls(
                                    selected_user,
                                    movie,
                                    (
                                        f"watchlist_"
                                        f"{watch_index}"
                                    )
                                )

                            if st.button(
                                "💔 Remove",
                                key=(
                                    f"watchlist_remove_"
                                    f"{selected_user}_"
                                    f"{movie_id}_"
                                    f"{watch_index}"
                                ),
                                use_container_width=True,
                            ):

                                remove_movie_from_watchlist(
                                    selected_user,
                                    movie_id
                                )

                                st.rerun()

    with tab6:

        st.subheader(
            f"⭐ My Ratings & Feedback - {selected_user}"
        )

        feedback = load_feedback()

        if feedback.empty:

            st.info(
                "You haven't rated or liked any movies yet."
            )

        else:

            user_feedback = feedback[
                feedback[
                    "user_id"
                ].astype(str)
                ==
                str(selected_user)
            ].copy()

            if user_feedback.empty:

                st.info(
                    "You haven't rated or liked any movies yet."
                )

            else:

                likes = int(
                    (
                        user_feedback[
                            "feedback"
                        ]
                        .astype(str)
                        .str.lower()
                        ==
                        "like"
                    ).sum()
                )

                dislikes = int(
                    (
                        user_feedback[
                            "feedback"
                        ]
                        .astype(str)
                        .str.lower()
                        ==
                        "dislike"
                    ).sum()
                )

                ratings = pd.to_numeric(
                    user_feedback[
                        "rating"
                    ],
                    errors="coerce"
                ).dropna()

                average = (
                    float(
                        ratings.mean()
                    )
                    if not ratings.empty
                    else None
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "👍 Likes",
                        likes
                    )

                with c2:
                    st.metric(
                        "👎 Dislikes",
                        dislikes
                    )

                with c3:
                    st.metric(
                        "⭐ Average Rating",
                        (
                            "N/A"
                            if average is None
                            else
                            f"{average:.1f}/5"
                        )
                    )

                table = (
                    user_feedback[
                        [
                            "title",
                            "feedback",
                            "rating",
                        ]
                    ]
                    .copy()
                    .rename(
                        columns={
                            "title": "Movie",
                            "feedback": "Feedback",
                            "rating": "Rating",
                        }
                    )
                )

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                )

    with tab7:

        st.subheader(
            "🏗️ OTT Recommendation Architecture"
        )

        st.code(
            """
OTT DATA
      ↓
DATA CLEANING
      ↓
FEATURE ENGINEERING
      ↓
CONTENT MODEL
      +
SVD COLLABORATIVE MODEL
      +
WATCH BEHAVIOUR
      +
SEARCH MODEL
      +
USER FEEDBACK
      ↓
FEEDBACK-AWARE BEHAVIOUR
      ↓
QUALITY SCORE
      ↓
HYBRID SCORE
      ↓
REMOVE WATCHED MOVIES
      ↓
GENRE DIVERSITY
      ↓
TOP N RECOMMENDATIONS
      ↓
TMDB
      ↓
REAL POSTER
      ↓
MOVIE CARD

USER SEARCH
      ↓
LOCAL DATASET + TMDB
      ↓
VIEW DETAILS
      ↓
WATCHLIST

USER FEEDBACK
      ↓
👍 LIKE
👎 DISLIKE
⭐ 1–5 RATING
      ↓
user_feedback.csv
      ↓
PERSONALIZATION PROFILE
      ↓
BETTER RECOMMENDATIONS

USER CLICKS
"WHY RECOMMENDED?"
      ↓
GROQ
      ↓
AI EXPLANATION
""",
            language="text",
        )

        st.subheader(
            "Hybrid Recommendation Formula"
        )

        st.write(
            "Final Score = "
            "0.30 × Content + "
            "0.25 × SVD + "
            "0.20 × Behaviour + "
            "0.10 × Search + "
            "0.15 × Quality"
        )

        st.subheader(
            "⚡ Performance Optimization"
        )

        st.write(
            "Data, watchlist, feedback, TMDB search results, "
            "poster lookups, and recommendation results are cached "
            "to reduce repeated work during Streamlit reruns."
        )

        st.subheader(
            "🎯 Personalization"
        )

        st.write(
            "Likes, dislikes, ratings, viewing behaviour, "
            "and search activity influence personalization."
        )

        st.subheader(
            "🔎 Search Integration"
        )

        st.write(
            "Search checks the local dataset and TMDB so that "
            "movies outside the local dataset can still be discovered."
        )

        st.subheader(
            "🎬 TMDB"
        )

        if TMDB_API_TOKEN:
            st.success(
                "✅ TMDB connection is active."
            )
        else:
            st.error(
                "❌ TMDB API token was not found."
            )

        st.caption(
            "Movie posters and external movie information are provided by TMDB."
        )

        st.subheader(
            "🤖 GenAI"
        )

        st.write(
            "Groq is called only when the user requests an explanation."
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "OTT Recommendation Agent | "
    "Content + Collaborative + Behaviour + "
    "Search + Quality + TMDB + Watchlist + "
    "Personalization + GenAI"
)
