import os
import requests
import logging

logger = logging.getLogger(__name__)

class IMDbAPIClient:
    def __init__(self):
        self.api_key = os.getenv("IMDB_API_KEY")
        self.dataset_id = os.getenv("IMDB_DATASET_ID")
        self.revision_id = os.getenv("IMDB_REVISION_ID")
        self.asset_id = os.getenv("IMDB_ASSET_ID")
        self.is_configured = bool(self.api_key and self.dataset_id)

    def is_available(self):
        return self.is_configured

    def _execute_graphql(self, query, variables=None):
        if not self.is_configured:
            return None
        # In a real environment, you would use AWS boto3 or directly hit AWS Data Exchange endpoint
        # with required AWS SigV4 Auth and x-api-key.
        url = "https://api.graphql.imdb.com/" # placeholder endpoint
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        try:
            resp = requests.post(
                url,
                json={"query": query, "variables": variables or {}},
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"IMDb API Error: {e}")
            return None

    def search_movies(self, query):
        if not self.is_available():
            logger.info("IMDb API not configured. Falling back to local search.")
            return None
        
        gql_query = """
        query SearchTitle($q: String!) {
          searchTitle(query: $q) {
            edges {
              node {
                id
                titleText { text }
                releaseYear { year }
                primaryImage { url }
              }
            }
          }
        }
        """
        data = self._execute_graphql(gql_query, {"q": query})
        # Parse data into structured format
        if data and "data" in data and "searchTitle" in data["data"]:
            results = []
            for edge in data["data"]["searchTitle"]["edges"]:
                node = edge["node"]
                results.append({
                    "imdb_id": node.get("id"),
                    "title": node.get("titleText", {}).get("text"),
                    "year": node.get("releaseYear", {}).get("year") if node.get("releaseYear") else None,
                    "poster_url": node.get("primaryImage", {}).get("url") if node.get("primaryImage") else None
                })
            return results
        return None

    def get_movie_details(self, imdb_id):
        if not self.is_available():
            return None

        gql_query = """
        query GetTitle($id: ID!) {
          title(id: $id) {
            id
            titleText { text }
            releaseYear { year }
            primaryImage { url }
            titleGenres { genres { genre { text } } }
            plot { plotText { plainText } }
            ratingsSummary { aggregateRating voteCount }
            titleRuntime { displayableProperty { value { plainText } } }
            certificate { rating }
            principalCredits {
                credits {
                    name { nameText { text } }
                }
            }
          }
        }
        """
        data = self._execute_graphql(gql_query, {"id": imdb_id})
        if data and "data" in data and "title" in data["data"]:
            node = data["data"]["title"]
            if not node:
                return None
            
            genres = [g["genre"]["text"] for g in node.get("titleGenres", {}).get("genres", [])]
            directors = [] # Would pull from specific credits edge
            cast = [] # Would pull from specific credits edge
            
            # Simplified parsing for demo structure
            if node.get("principalCredits"):
                for pc in node["principalCredits"]:
                    for credit in pc.get("credits", []):
                        cast.append(credit["name"]["nameText"]["text"])
                        
            return {
                "imdb_id": node.get("id"),
                "title": node.get("titleText", {}).get("text"),
                "year": node.get("releaseYear", {}).get("year") if node.get("releaseYear") else None,
                "poster_url": node.get("primaryImage", {}).get("url") if node.get("primaryImage") else None,
                "genres": genres,
                "plot": node.get("plot", {}).get("plotText", {}).get("plainText"),
                "rating": node.get("ratingsSummary", {}).get("aggregateRating"),
                "vote_count": node.get("ratingsSummary", {}).get("voteCount"),
                "runtime": node.get("titleRuntime", {}).get("displayableProperty", {}).get("value", {}).get("plainText"),
                "certificate": node.get("certificate", {}).get("rating"),
                "cast": ", ".join(cast[:5]), # Just getting a few names
                "director": "Various" # Placeholder unless structured director edge available
            }
        return None
