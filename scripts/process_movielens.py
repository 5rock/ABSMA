import os
import pandas as pd
import json
import ast

def parse_json_safely(x):
    try:
        # Some fields use single quotes like python dicts, ast.literal_eval is safer than json.loads
        return ast.literal_eval(x)
    except:
        return []

def get_director(crew_str):
    crew = parse_json_safely(crew_str)
    directors = [c['name'] for c in crew if c.get('job') == 'Director']
    return ", ".join(directors) if directors else "Unknown"

def get_cast(cast_str):
    cast = parse_json_safely(cast_str)
    # Get top 3 cast members
    top_cast = [c['name'] for c in cast[:3]]
    return ", ".join(top_cast) if top_cast else "Unknown"

def get_genres(genres_str):
    genres = parse_json_safely(genres_str)
    genre_names = [g['name'] for g in genres]
    return "|".join(genre_names) if genre_names else "Unknown"

def get_edge_rating(vote_avg):
    try:
        v = float(vote_avg)
        if v >= 8.0: return "A+"
        if v >= 7.0: return "A"
        if v >= 6.0: return "B+"
        if v >= 5.0: return "B"
        return "C"
    except:
        return "B"

def get_age_category(adult_flag):
    if str(adult_flag).lower() == 'true':
        return '18+'
    return 'Teen'

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    metadata_in = os.path.join(project_root, "data", "datasets", "movies_metadata.csv")
    credits_in = os.path.join(project_root, "data", "datasets", "credits.csv")
    out_file = os.path.join(project_root, "data", "movie_metadata.csv")

    print("Loading datasets...")
    # Read CSVs (low_memory=False to avoid DtypeWarning)
    df_meta = pd.read_csv(metadata_in, low_memory=False)
    df_credits = pd.read_csv(credits_in, low_memory=False)

    print(f"Original movies count: {len(df_meta)}")
    
    # Filter valid rows (must have id and title)
    df_meta = df_meta[df_meta['id'].notnull() & df_meta['title'].notnull()]
    
    # Convert ID to numeric for merging
    df_meta['id'] = pd.to_numeric(df_meta['id'], errors='coerce')
    df_credits['id'] = pd.to_numeric(df_credits['id'], errors='coerce')
    
    df_meta = df_meta.dropna(subset=['id'])
    df_meta['id'] = df_meta['id'].astype(int)
    df_credits = df_credits.dropna(subset=['id'])
    df_credits['id'] = df_credits['id'].astype(int)

    # Merge meta and credits on id
    print("Merging metadata and credits...")
    df = pd.merge(df_meta, df_credits, on='id', how='inner')

    # Convert vote_count to numeric and filter to get popular movies
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce').fillna(0)
    # Keep top 5000 movies by vote count to ensure high quality posters and data
    df = df.sort_values(by='vote_count', ascending=False).head(5000)

    print(f"Processing {len(df)} movies...")
    
    # Process columns to match application format
    # Columns needed: movie_id, title, year, genres, age_category, edge_rating, description, director, cast, runtime, poster_url
    
    output = []
    
    for idx, row in df.iterrows():
        movie_id = row['id']
        title = row['title']
        
        # Year
        try:
            year = str(row['release_date'])[:4]
            if year == 'nan': year = "Unknown"
        except:
            year = "Unknown"
            
        # Genres
        genres = get_genres(row['genres'])
        
        # Age Category
        age_category = get_age_category(row['adult'])
        
        # Edge Rating
        edge_rating = get_edge_rating(row['vote_average'])
        
        # Description
        description = row['overview'] if pd.notna(row['overview']) else "No description available."
        
        # Director and Cast
        director = get_director(row['crew'])
        cast = get_cast(row['cast'])
        
        # Runtime
        try:
            rt = float(row['runtime'])
            runtime = f"{int(rt)} min" if pd.notna(rt) else "Unknown"
        except:
            runtime = "Unknown"
            
        # Poster URL
        poster_path = row['poster_path']
        if pd.notna(poster_path) and poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        else:
            poster_url = "https://via.placeholder.com/500x750?text=No+Poster"
            
        output.append({
            "movie_id": movie_id,
            "title": title,
            "year": year,
            "genres": genres,
            "age_category": age_category,
            "edge_rating": edge_rating,
            "description": description,
            "director": director,
            "cast": cast,
            "runtime": runtime,
            "poster_url": poster_url
        })

    df_out = pd.DataFrame(output)
    
    # Ensure no duplicates by ID
    df_out = df_out.drop_duplicates(subset=['movie_id'])
    
    df_out.to_csv(out_file, index=False, encoding='utf-8')
    print(f"Successfully generated {len(df_out)} movies in {out_file}")

if __name__ == "__main__":
    main()
