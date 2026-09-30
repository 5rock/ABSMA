from functools import wraps
from flask import request, jsonify, g
import jwt
import os
from database import SessionLocal
from models.user import User

def get_token_from_request():
    token = request.cookies.get('access_token')
    if token:
        return token
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        return auth_header.split(" ")[1]
    return None

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            return jsonify({"error": "Missing access token"}), 401
        
        try:
            secret = os.getenv("JWT_SECRET_KEY")
            data = jwt.decode(token, secret, algorithms=["HS256"])
            db = SessionLocal()
            try:
                user = db.query(User).filter(User.id == int(data["sub"])).first()
                if not user or not user.is_active:
                    return jsonify({"error": "User not found or inactive"}), 401
                g.current_user = user
            finally:
                db.close()
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
            
        return f(*args, **kwargs)
    return decorated

# Optional decorator if we want to pass current user, but it's on `g.current_user`
