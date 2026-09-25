import os
import pandas as pd
import json
import logging
from services.imdb_api import IMDbAPIClient

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "imdb_cache")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "movie_metadata.csv")

class MovieService:
    def __init__(self):
        self.imdb_client = IMDbAPIClient()
        self._local_movies = []
        self._load_local_data()
        
    def _normalize_title(self, title):
        if not title:
            return ""
        return " ".join(str(title).lower().split())
        
    def _normalize_genres(self, genres):
        if not genres:
            return []
        if isinstance(genres, str):
            genres = genres.split("|")
        seen = set()
        out = []
        for g in genres:
            g = g.strip()
            if g and g.lower() not in seen:
                seen.add(g.lower())
                out.append(g)
        return out
        
    def _normalize_cast(self, cast_str):
        if not cast_str or str(cast_str) == "nan":
            return ""
        cast_list = str(cast_str).split(",")
        seen = set()
        out = []
        for c in cast_list:
            c = c.strip()
            if c and c.lower() not in seen:
                seen.add(c.lower())
                out.append(c)
        return ", ".join(out)

    def _load_local_data(self):
        try:
            df = pd.read_csv(METADATA_PATH)
            movies = []
            seen_keys = set()
            
            for _, row in df.iterrows():
                title = str(row.get("title", ""))
                year = str(row.get("year", ""))
                if year == "nan": year = ""
                
                # Check imdb_id if it exists
                imdb_id = str(row.get("imdb_id", ""))
                if imdb_id and imdb_id != "nan":
                    unique_key = imdb_id
                else:
                    unique_key = f"{self._normalize_title(title)}_{year}"
                
                if unique_key in seen_keys:
                    continue
                seen_keys.add(unique_key)
                
                movies.append({
                    "imdb_id": imdb_id if imdb_id != "nan" else None,
                    "id": str(row.get("movie_id", "")),
                    "title": title,
                    "year": year,
                    "genres": self._normalize_genres(row.get("genres", "")),
                    "age_category": str(row.get("age_category", "")).replace("nan", ""),
                    "edge_rating": str(row.get("edge_rating", "")).replace("nan", ""),
                    "description": str(row.get("description", "")).replace("nan", ""),
                    "director": str(row.get("director", "")).replace("nan", ""),
                    "cast": self._normalize_cast(row.get("cast", "")),
                    "runtime": str(row.get("runtime", "")).replace("nan", ""),
                    "poster_url": str(row.get("poster_url", "")).replace("nan", ""),
                    "imdb_rating": None,
                    "vote_count": None,
                    "certificate": None
                })
            self._local_movies = movies
        except Exception as e:
            logger.error(f"Failed to load local metadata: {e}")
            self._local_movies = []

    def get_all_movies(self):
        return self._local_movies
        
    def get_movie_by_id(self, movie_id):
        movie_id = str(movie_id)
        for m in self._local_movies:
            if str(m["id"]) == movie_id or str(m.get("imdb_id", "")) == movie_id:
                if m.get("imdb_id") and self.imdb_client.is_available():
                    imdb_data = self._get_cached_imdb_data(m["imdb_id"])
                    if imdb_data:
                        self._merge_imdb_data(m, imdb_data)
                return m
        
        if movie_id.startswith("tt") and self.imdb_client.is_available():
            imdb_data = self._get_cached_imdb_data(movie_id)
            if imdb_data:
                return imdb_data
                
        return None

    def search_movies(self, query):
        if not query:
            return self.get_all_movies()
            
        q = query.lower()
        results = []
        
        for m in self._local_movies:
            match = False
            if q in str(m.get("title", "")).lower(): match = True
            elif q in str(m.get("description", "")).lower(): match = True
            elif q in str(m.get("director", "")).lower(): match = True
            elif q in str(m.get("cast", "")).lower(): match = True
            elif any(q in g.lower() for g in m.get("genres", [])): match = True
            
            if match:
                results.append(m)
                
        if self.imdb_client.is_available():
            imdb_results = self.imdb_client.search_movies(query)
            if imdb_results:
                local_imdb_ids = {m["imdb_id"] for m in results if m.get("imdb_id")}
                local_keys = {f"{self._normalize_title(m['title'])}_{m['year']}" for m in results}
                
                for im in imdb_results:
                    key = f"{self._normalize_title(im['title'])}_{im['year']}"
                    if im.get("imdb_id") in local_imdb_ids or key in local_keys:
                        continue
                    results.append(im)
                    
        return results
        
    def filter_movies(self, genres, ages, ratings, base_movies=None):
        if base_movies is None:
            base_movies = self._local_movies
            
        filtered = []
        for m in base_movies:
            # Genre logic (OR within genre)
            if genres:
                g_lower = [g.lower() for g in genres]
                movie_g_lower = [mg.lower() for mg in m.get("genres", [])]
                if not any(g in movie_g_lower for g in g_lower):
                    continue
            
            # Age logic (OR within ages)
            if ages:
                if m.get("age_category") not in ages:
                    continue
                    
            # Rating logic (OR within ratings)
            if ratings:
                if m.get("edge_rating") not in ratings:
                    continue
                    
            filtered.append(m)
        return filtered
        
    def _get_cached_imdb_data(self, imdb_id):
        cache_file = os.path.join(CACHE_DIR, f"{imdb_id}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
                
        data = self.imdb_client.get_movie_details(imdb_id)
        if data:
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        return None
        
    def _merge_imdb_data(self, local_movie, imdb_data):
        if imdb_data.get("title"): local_movie["title"] = imdb_data["title"]
        if imdb_data.get("year"): local_movie["year"] = imdb_data["year"]
        if imdb_data.get("genres"): local_movie["genres"] = imdb_data["genres"]
        if imdb_data.get("plot"): local_movie["description"] = imdb_data["plot"]
        if imdb_data.get("poster_url"): local_movie["poster_url"] = imdb_data["poster_url"]
        if imdb_data.get("rating"): local_movie["imdb_rating"] = imdb_data["rating"]
        if imdb_data.get("vote_count"): local_movie["vote_count"] = imdb_data["vote_count"]
        if imdb_data.get("runtime"): local_movie["runtime"] = imdb_data["runtime"]
        if imdb_data.get("certificate"): local_movie["certificate"] = imdb_data["certificate"]
        if imdb_data.get("director"): local_movie["director"] = imdb_data["director"]
        if imdb_data.get("cast"): local_movie["cast"] = imdb_data["cast"]
