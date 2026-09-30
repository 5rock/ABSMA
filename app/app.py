import os
import json
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS
from dotenv import load_dotenv

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load environment variables
load_dotenv()

from src.absa_pipeline import ABSAPipeline
from services.movie_service import MovieService
from database import init_db, get_db, SessionLocal, engine
from auth.routes import auth_bp
from auth.middleware import token_required
from models.analysis import AnalysisHistory
from sqlalchemy import text

app = Flask(__name__)
# Restrict CORS to specific origins or avoid * with credentials
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000", "http://localhost:5010", "http://127.0.0.1:5010"])

# Register Blueprints
app.register_blueprint(auth_bp)

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

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/signup")
def signup_page():
    return render_template("signup.html")

# ─────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health_check():
    db_status = "unhealthy"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_status = "healthy"
    except Exception as e:
        db_status = f"error: {str(e)}"
        
    return jsonify({
        "status": "ok" if db_status == "healthy" else "error",
        "database": db_status
    }), 200 if db_status == "healthy" else 503

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
@token_required
def analyze_review():
    data = request.get_json()
    if not data or "review_text" not in data:
        return jsonify({"error": "Missing 'review_text'"}), 400
    
    text_content = data["review_text"].strip()
    movie_title = data.get("movie_title", "").strip() or "Unknown Movie"
    
    if not text_content:
        return jsonify({"error": "Empty review text"}), 400
        
    pipeline = get_pipeline()
    try:
        results = pipeline.analyze(text_content)
        
        # Save to analysis history
        db = SessionLocal()
        try:
            history = AnalysisHistory(
                user_id=g.current_user.id,
                movie_title=movie_title,
                review_text=text_content,
                result=results
            )
            db.add(history)
            db.commit()
        finally:
            db.close()
            
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Error during analysis: {e}")
        return jsonify({"error": str(e)}), 500

from sqlalchemy import or_, and_, extract
from datetime import datetime, timezone, timedelta

@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    return render_template("dashboard.html")

@app.route("/api/dashboard", methods=["GET"])
@token_required
def get_dashboard_data():
    db = SessionLocal()
    try:
        month = request.args.get("month", type=int)
        year = request.args.get("year", type=int)
        
        now = datetime.now(timezone.utc)
        if not month or not year:
            month = now.month
            year = now.year
            
        # Get all analyses for the user in the selected month and year
        # Fetching all to calculate JSON stats in memory for cross-db compatibility
        query = db.query(AnalysisHistory).filter(
            AnalysisHistory.user_id == g.current_user.id,
            extract('month', AnalysisHistory.created_at) == month,
            extract('year', AnalysisHistory.created_at) == year
        ).order_by(AnalysisHistory.created_at.desc())
        
        records = query.all()
        
        movies_analyzed = len(set([r.movie_title for r in records if r.movie_title]))
        reviews_analyzed = len(records)
        
        pos = 0
        neg = 0
        neu = 0
        
        aspect_counts = {}
        weekly_counts = [0, 0, 0, 0, 0] 
        recent_analyses = []
        
        for idx, r in enumerate(records):
            if idx < 5:
                recent_analyses.append({
                    "id": r.id,
                    "movie_title": r.movie_title,
                    "review_text": r.review_text,
                    "result": r.result,
                    "analyzed_at": r.created_at.isoformat() if r.created_at else None
                })
                
            day = r.created_at.day
            week_idx = (day - 1) // 7
            if week_idx > 4: week_idx = 4
            weekly_counts[week_idx] += 1
                
            if r.result and "aspects" in r.result:
                for aspect in r.result["aspects"]:
                    sentiment = aspect.get("sentiment")
                    category = aspect.get("category")
                    
                    if sentiment == "positive": pos += 1
                    elif sentiment == "negative": neg += 1
                    elif sentiment == "neutral": neu += 1
                    
                    if category:
                        aspect_counts[category] = aspect_counts.get(category, 0) + 1
                        
        top_aspects = sorted([{"category": k, "count": v} for k, v in aspect_counts.items()], key=lambda x: x["count"], reverse=True)[:5]
        
        monthly_activity = []
        for w in range(5):
            if weekly_counts[w] > 0 or w < 4:
                monthly_activity.append({"label": f"Week {w+1}", "count": weekly_counts[w]})
                
        return jsonify({
            "success": True,
            "stats": {
                "movies_analyzed": movies_analyzed,
                "reviews_analyzed": reviews_analyzed,
                "positive": pos,
                "neutral": neu,
                "negative": neg
            },
            "top_aspects": top_aspects,
            "monthly_activity": monthly_activity,
            "recent_analyses": recent_analyses
        })
    finally:
        db.close()

@app.route("/history", methods=["GET"])
def history_page():
    return render_template("history.html")

@app.route("/api/history", methods=["GET"])
@token_required
def get_analysis_history():
    db = SessionLocal()
    try:
        query = db.query(AnalysisHistory).filter(AnalysisHistory.user_id == g.current_user.id)
        
        # Search
        search = request.args.get("search", "")
        if search:
            search_filter = f"%{search}%"
            query = query.filter(
                or_(
                    AnalysisHistory.movie_title.ilike(search_filter),
                    AnalysisHistory.review_text.ilike(search_filter)
                )
            )
            
        # Date filtering
        period = request.args.get("period", "")
        month = request.args.get("month", type=int)
        year = request.args.get("year", type=int)
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        
        now = datetime.now(timezone.utc)
        
        if period == "today":
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(AnalysisHistory.created_at >= start_of_day)
        elif period == "week":
            start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(AnalysisHistory.created_at >= start_of_week)
        elif period == "month":
            start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(AnalysisHistory.created_at >= start_of_month)
        elif period == "last_month":
            first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            last_month_end = first_of_this_month - timedelta(seconds=1)
            start_of_last_month = last_month_end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            query = query.filter(AnalysisHistory.created_at >= start_of_last_month, AnalysisHistory.created_at <= last_month_end)
        elif month and year:
            # Note: extract with sqlite might not work exactly this way but we'll use it
            # SQLite extract works differently, so let's use string manipulation if extract fails
            query = query.filter(extract('month', AnalysisHistory.created_at) == month, extract('year', AnalysisHistory.created_at) == year)
        elif start_date and end_date:
            try:
                start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                end = datetime.strptime(end_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                query = query.filter(AnalysisHistory.created_at >= start, AnalysisHistory.created_at <= end)
            except ValueError:
                pass

        # Pagination
        page = request.args.get("page", 1, type=int)
        limit = request.args.get("limit", 20, type=int)
        
        total = query.count()
        total_pages = (total + limit - 1) // limit
        
        history = query.order_by(AnalysisHistory.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
        
        analyses = []
        for h in history:
            analyses.append({
                "id": h.id,
                "movie_title": h.movie_title,
                "review_text": h.review_text,
                "result": h.result,
                "analyzed_at": h.created_at.isoformat() if h.created_at else None
            })
            
        return jsonify({
            "success": True,
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": total_pages,
            "analyses": analyses
        })
    finally:
        db.close()

@app.route("/api/history/<int:history_id>", methods=["DELETE"])
@token_required
def delete_history(history_id):
    db = SessionLocal()
    try:
        record = db.query(AnalysisHistory).filter(AnalysisHistory.id == history_id, AnalysisHistory.user_id == g.current_user.id).first()
        if not record:
            return jsonify({"error": "Record not found or unauthorized"}), 404
            
        db.delete(record)
        db.commit()
        return jsonify({"success": True, "message": "Deleted successfully"})
    finally:
        db.close()

if __name__ == "__main__":
    print("=" * 60)
    print("  MovieLens ABSA — Flask Backend")
    print("=" * 60)
    print(f"  Project root : {project_root}")
    print("  Frontend URL : http://localhost:5000")
    print("=" * 60)
    
    # Initialize DB (creates tables if not exists)
    init_db()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
