from flask import Blueprint, request, jsonify, make_response, current_app
from database import get_db
from models.user import User
import jwt
import os
from datetime import datetime, timedelta, timezone
from auth.middleware import token_required, g

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "message": "Invalid JSON or missing body"}), 400
            
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        
        if not name or not email or not password:
            return jsonify({"success": False, "message": "Missing required fields"}), 400
            
        db = next(get_db())
        if db.query(User).filter(User.email == email).first():
            return jsonify({"success": False, "message": "An account with this email already exists."}), 409
            
        new_user = User(name=name, email=email)
        new_user.set_password(password)
        
        db.add(new_user)
        db.commit()
        
        return jsonify({"success": True, "message": "Registration successful"}), 201
    except Exception as e:
        current_app.logger.error(f"Registration error: {e}")
        return jsonify({"success": False, "message": f"Internal server error: {e}"}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request"}), 400
        
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({"error": "Missing required fields"}), 400
        
    db = next(get_db())
    user = db.query(User).filter(User.email == email).first()
    
    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid credentials"}), 401
        
    if not user.is_active:
        return jsonify({"error": "Account is disabled"}), 401
        
    user.last_login = datetime.now(timezone.utc)
    db.commit()
    
    secret = os.getenv("JWT_SECRET_KEY")
    expires_in = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))
    
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(seconds=expires_in)
    }
    
    token = jwt.encode(payload, secret, algorithm="HS256")
    
    # Store token in HttpOnly cookie
    response = make_response(jsonify({
        "message": "Login successful",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }))
    
    # Secure in production, False for dev
    is_secure = os.getenv("FLASK_ENV") == "production"
    
    response.set_cookie(
        'access_token',
        token,
        httponly=True,
        secure=is_secure,
        samesite='Lax',
        max_age=expires_in
    )
    
    return response

@auth_bp.route('/logout', methods=['POST'])
def logout():
    response = make_response(jsonify({"message": "Logged out successfully"}))
    response.delete_cookie('access_token')
    return response

@auth_bp.route('/me', methods=['GET'])
@token_required
def get_me():
    user = g.current_user
    return jsonify({
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "created_at": user.created_at.isoformat() if user.created_at else None
    })
