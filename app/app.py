import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

from src.absa_pipeline import ABSAPipeline
from services.movie_service import MovieService

app = Flask(__name__)
CORS(app)

# Initialize Movie Service (loads metadata, handles deduplication & IMDb fallback)
movie_service = MovieService()

# ─────────────────────────────────────────────────────────────
# LAZY-LOAD ABSA PIPELINE (only when first /analyze request)
# ─────────────────────────────────────────────────────────────
_absa_pipeline = None

def get_pipeline() -> ABSAPipeline:
    """Lazy-load the ABSA pipeline to avoid slow startup."""
    global _absa_pipeline
    if _absa_pipeline is None:
        print("[SERVER] Loading ABSA pipeline...")
        _absa_pipeline = ABSAPipeline(use_transformer_ner=True)
        print("[SERVER] ABSA pipeline ready.")
    return _absa_pipeline

# ─────────────────────────────────────────────────────────────
# FRONTEND ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/movies")
def movies_page():
    return render_template("movies.html")

@app.route("/movie/<movie_id>")
def movie_detail_page(movie_id):
    movie = movie_service.get_movie_by_id(movie_id)
    if not movie:
        return "Movie not found", 404
    return render_template("movie_detail.html", movie=movie)

@app.route("/analyze")
def analyze_page():
    movie_id = request.args.get("movie_id")
    movie = None
    if movie_id:
        movie = movie_service.get_movie_by_id(movie_id)
    return render_template("analyze.html", movie=movie)

@app.route("/about")
def about_page():
    return render_template("about.html")

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/api/movies", methods=["GET"])
def get_movies():
    movies = movie_service.get_all_movies()
    return jsonify({"movies": movies, "total": len(movies)})

@app.route("/api/movies/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").strip()
    movies = movie_service.search_movies(query)
    return jsonify({"movies": movies, "total": len(movies), "query": query})

@app.route("/api/movies/filter", methods=["GET"])
def filter_movies():
    genres = request.args.getlist("genre")
    ages = request.args.getlist("age")
    ratings = request.args.getlist("rating")
    
    # Optional search query along with filters
    query = request.args.get("q", "").strip()
    
    if query:
        base_movies = movie_service.search_movies(query)
        filtered = movie_service.filter_movies(genres, ages, ratings, base_movies=base_movies)
    else:
        filtered = movie_service.filter_movies(genres, ages, ratings)
        
    return jsonify({"movies": filtered, "total": len(filtered)})

@app.route("/api/movie/<movie_id>", methods=["GET"])
def get_movie(movie_id):
    movie = movie_service.get_movie_by_id(movie_id)
    if movie:
        return jsonify(movie)
    return jsonify({"error": "Movie not found"}), 404

@app.route("/api/analyze", methods=["POST"])
def analyze_review():
    data = request.get_json()
    if not data or "review_text" not in data:
        return jsonify({"error": "Missing 'review_text'"}), 400
    
    text = data["review_text"].strip()
    if not text:
        return jsonify({"error": "Empty review text"}), 400
        
    pipeline = get_pipeline()
    try:
        results = pipeline.analyze_review(text)
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("=" * 60)
    print("  MovieLens ABSA — Flask Backend")
    print("=" * 60)
    print(f"  Project root : {project_root}")
    print("  Frontend URL : http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
