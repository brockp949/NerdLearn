"""
API Gateway - Central entry point for all NerdLearn services

Responsibilities:
1. Authentication & authorization
2. Route requests to appropriate microservices
3. CORS handling
4. Error handling
5. Request/response logging
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select
from datetime import datetime, timedelta
from typing import Optional
import httpx

from auth import (
    Token, UserCreate, UserLogin, UserResponse,
    create_access_token, create_refresh_token,
    get_password_hash, verify_password,
    validate_password, validate_email,
    get_current_active_user, TokenData
)

# ============================================================================
# APP CONFIGURATION
# ============================================================================

app = FastAPI(
    title="NerdLearn API Gateway",
    description="Central API gateway for all NerdLearn services",
    version="0.1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service URLs
SCHEDULER_URL = "http://localhost:8001"
TELEMETRY_URL = "http://localhost:8002"
INFERENCE_URL = "http://localhost:8003"
CONTENT_URL = "http://localhost:8004"

# HTTP client for service calls
http_client = httpx.AsyncClient()


# ============================================================================
# DATABASE SETUP (Simple in-memory for now)
# ============================================================================

# In-memory user store (TODO: Replace with actual database)
users_db = {}

class User:
    """Simple user model"""
    def __init__(self, id: str, email: str, username: str, password_hash: str, role: str = "STUDENT"):
        self.id = id
        self.email = email
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self.created_at = datetime.utcnow()


def get_user_by_email(email: str) -> Optional[User]:
    """Get user by email"""
    for user in users_db.values():
        if user.email == email:
            return user
    return None


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID"""
    return users_db.get(user_id)


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.post("/api/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """Register a new user"""

    # Validate email
    if not validate_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid email format"
        )

    # Validate password
    if not validate_password(user_data.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters and contain letters and numbers"
        )

    # Check if user exists
    if get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user_id = f"user_{len(users_db) + 1}"
    password_hash = get_password_hash(user_data.password)

    user = User(
        id=user_id,
        email=user_data.email,
        username=user_data.username,
        password_hash=password_hash
    )

    users_db[user_id] = user

    # Create learner profile in database (TODO: Call database service)

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role,
        created_at=user.created_at
    )


@app.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get JWT tokens"""

    # Get user
    user = get_user_by_email(form_data.username)  # username field contains email

    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create tokens
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.id, "email": user.email}
    )

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: TokenData = Depends(get_current_active_user)):
    """Get current user information"""

    user = get_user_by_id(current_user.user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        role=user.role,
        created_at=user.created_at
    )


@app.post("/api/auth/logout")
async def logout(current_user: TokenData = Depends(get_current_active_user)):
    """Logout (client should discard tokens)"""
    # In a real app, you might want to blacklist the token
    return {"message": "Successfully logged out"}


# ============================================================================
# SCHEDULER SERVICE PROXY
# ============================================================================

@app.post("/api/scheduler/review")
async def process_review(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Proxy to scheduler service - process review"""
    try:
        response = await http_client.post(
            f"{SCHEDULER_URL}/review",
            json={**request, "learner_id": current_user.user_id}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Scheduler service unavailable: {str(e)}"
        )


@app.get("/api/scheduler/due")
async def get_due_cards(
    limit: int = 20,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get cards due for review"""
    try:
        response = await http_client.get(
            f"{SCHEDULER_URL}/due/{current_user.user_id}",
            params={"limit": limit}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Scheduler service unavailable: {str(e)}"
        )


# ============================================================================
# INFERENCE SERVICE PROXY
# ============================================================================

@app.post("/api/inference/predict")
async def predict_performance(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Predict performance on concept"""
    try:
        response = await http_client.post(
            f"{INFERENCE_URL}/predict",
            json={**request, "learner_id": current_user.user_id}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {str(e)}"
        )


@app.post("/api/inference/zpd/assess")
async def assess_zpd(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Assess ZPD state"""
    try:
        response = await http_client.post(
            f"{INFERENCE_URL}/zpd/assess",
            json={**request, "learner_id": current_user.user_id}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {str(e)}"
        )


@app.post("/api/inference/recommend")
async def get_recommendations(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get adaptive recommendations"""
    try:
        response = await http_client.post(
            f"{INFERENCE_URL}/recommend",
            json={**request, "learner_id": current_user.user_id}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference service unavailable: {str(e)}"
        )


# ============================================================================
# TELEMETRY SERVICE PROXY
# ============================================================================

@app.post("/api/telemetry/event")
async def ingest_event(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Ingest telemetry event"""
    try:
        response = await http_client.post(
            f"{TELEMETRY_URL}/event",
            json={**request, "user_id": current_user.user_id}
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Telemetry service unavailable: {str(e)}"
        )


@app.get("/api/telemetry/engagement/{session_id}")
async def get_engagement(
    session_id: str,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Get engagement score"""
    try:
        response = await http_client.get(
            f"{TELEMETRY_URL}/analysis/engagement/{current_user.user_id}/{session_id}"
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Telemetry service unavailable: {str(e)}"
        )


# ============================================================================
# CONTENT SERVICE PROXY
# ============================================================================

@app.post("/api/content/analyze/text")
async def analyze_text(
    request: dict,
    current_user: TokenData = Depends(get_current_active_user)
):
    """Analyze text content"""
    try:
        response = await http_client.post(
            f"{CONTENT_URL}/analyze/text",
            params=request
        )
        return response.json()
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Content service unavailable: {str(e)}"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "NerdLearn API Gateway",
        "status": "operational",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """Check health of all services"""
    services = {
        "gateway": "healthy",
        "scheduler": "unknown",
        "telemetry": "unknown",
        "inference": "unknown",
        "content": "unknown"
    }

    # Check each service
    try:
        response = await http_client.get(f"{SCHEDULER_URL}/")
        services["scheduler"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["scheduler"] = "unhealthy"

    try:
        response = await http_client.get(f"{TELEMETRY_URL}/")
        services["telemetry"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["telemetry"] = "unhealthy"

    try:
        response = await http_client.get(f"{INFERENCE_URL}/")
        services["inference"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["inference"] = "unhealthy"

    try:
        response = await http_client.get(f"{CONTENT_URL}/")
        services["content"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        services["content"] = "unhealthy"

    return services


# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    print("✅ API Gateway started")
    print("🔐 Authentication enabled")
    print("🌐 CORS configured for http://localhost:3000")


@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
    print("🛑 API Gateway stopped")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
