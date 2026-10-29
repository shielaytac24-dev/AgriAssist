import os
import uuid
import jwt

from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Header,
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from werkzeug.security import check_password_hash, generate_password_hash
from dotenv import load_dotenv

from ml.predictor import predict_image


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SECRET = os.getenv(
    "API_SECRET_KEY",
    os.getenv(
        "SECRET_KEY",
        "change-this-secret-key-before-deployment"
    )
)

ALGO = "HS256"
TOKEN_HOURS = 8


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AgriAssist REST API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

def _cors_origins():
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"
    )
    origins = [
        origin.strip()
        for origin in raw.split(",")
        if origin.strip() and origin.strip() != "*"
    ]
    if not origins:
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STATIC UPLOADS
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def db():
    """
    Create and return a PostgreSQL database connection.
    Uses DATABASE_URL if available.
    Otherwise uses individual DB_* environment variables.
    """

    url = os.getenv("DATABASE_URL")

    if url:
        return psycopg2.connect(url)

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME", "agriassist_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres"),
    )


# ============================================================
# DATABASE QUERY HELPER
# ============================================================

def q(
    sql,
    params=(),
    one=False,
    commit=False
):
    """
    Execute a database query.

    one=True:
        Return one row.

    one=False:
        Return all rows.

    commit=True:
        Commit changes.

    This version correctly supports queries
    with RETURNING clauses.
    """

    conn = db()

    try:
        with conn.cursor(
            cursor_factory=RealDictCursor
        ) as cur:

            cur.execute(sql, params)

            # SELECT or INSERT/UPDATE/DELETE ... RETURNING
            if cur.description is not None:

                if one:
                    result = cur.fetchone()
                else:
                    result = cur.fetchall()

                if commit:
                    conn.commit()

                return result

            # INSERT/UPDATE/DELETE without RETURNING
            if commit:
                conn.commit()

            return None

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# JWT TOKEN
# ============================================================

def make_token(user):
    now = datetime.now(timezone.utc)

    payload = {
        "sub": str(user["id"]),
        "role": user["user_type"],
        "iat": now,
        "exp": now + timedelta(hours=TOKEN_HOURS),
    }

    return jwt.encode(
        payload,
        SECRET,
        algorithm=ALGO
    )


# ============================================================
# CURRENT USER
# ============================================================

def current_user(
    authorization: Optional[str] = Header(None)
):

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header"
        )

    token = authorization.split(
        " ",
        1
    )[1].strip()

    try:

        payload = jwt.decode(
            token,
            SECRET,
            algorithms=[ALGO]
        )

        uid = int(payload["sub"])

    except (
        jwt.PyJWTError,
        ValueError,
        KeyError
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    user = q(
        """
        SELECT
            id,
            full_name,
            username,
            email,
            user_type,
            location,
            avatar_filename
        FROM users
        WHERE id = %s
        """,
        (uid,),
        one=True
    )

    if not user:
        raise HTTPException(
            status_code=401,
            detail="User not found"
        )

    return user


# ============================================================
# ADMIN USER
# ============================================================

def admin_user(
    user=Depends(current_user)
):

    if user["user_type"] != "Administrator":

        raise HTTPException(
            status_code=403,
            detail="Administrator access required"
        )

    return user


def _safe_user(user):
    return {
        "id": user["id"],
        "full_name": user["full_name"],
        "username": user["username"],
        "email": user["email"],
        "user_type": user["user_type"],
        "location": user.get("location"),
        "avatar_url": (
            f"/uploads/{user['avatar_filename']}"
            if user.get("avatar_filename")
            else None
        ),
    }


def ensure_schema():
    statements = [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_filename VARCHAR(255)",
        "ALTER TABLE community_posts ADD COLUMN IF NOT EXISTS category VARCHAR(40) DEFAULT 'Experiences'",
        "ALTER TABLE tips ADD COLUMN IF NOT EXISTS category VARCHAR(40)",
    ]
    for sql in statements:
        try:
            q(sql, commit=True)
        except Exception:
            pass


ensure_schema()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        q(
            "SELECT 1",
            one=True
        )

        model = os.getenv(
            "AGRIASSIST_MODEL",
            os.path.join(
                ROOT,
                "ml",
                "weights",
                "best.pt"
            )
        )

        return {
            "status": "ok",
            "database": "postgresql",
            "cnn_model_available": os.path.exists(model)
        }

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "database": str(e)
            }
        )


# ============================================================
# REGISTER
# ============================================================

@app.post("/api/auth/register")
def register(payload: dict):

    required = [
        "full_name",
        "username",
        "email",
        "user_type",
        "location",
        "password"
    ]

    # Check required fields
    for key in required:

        if not str(
            payload.get(key, "")
        ).strip():

            raise HTTPException(
                status_code=400,
                detail="Complete all required fields"
            )

    password = str(
        payload.get("password", "")
    )

    if len(password) < 8:

        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters"
        )

    allowed_roles = {
        "Farmer",
        "Agricultural Worker",
        "Community Member"
    }

    if payload["user_type"] not in allowed_roles:

        raise HTTPException(
            status_code=400,
            detail="Invalid user type"
        )

    username = str(
        payload["username"]
    ).strip()

    email = str(
        payload["email"]
    ).strip().lower()

    # Check duplicate account
    existing = q(
        """
        SELECT id
        FROM users
        WHERE username = %s
           OR email = %s
        LIMIT 1
        """,
        (
            username,
            email
        ),
        one=True
    )

    if existing:

        raise HTTPException(
            status_code=409,
            detail="Username or email already registered"
        )

    password_hash = generate_password_hash(
        password
    )

    user = q(
        """
        INSERT INTO users (
            full_name,
            username,
            email,
            password_hash,
            user_type,
            location
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING
            id,
            full_name,
            username,
            email,
            user_type,
            location
        """,
        (
            str(payload["full_name"]).strip(),
            username,
            email,
            password_hash,
            payload["user_type"],
            str(payload["location"]).strip()
        ),
        one=True,
        commit=True
    )

    return {
        "message": "Account created",
        "user": user
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(payload: dict):

    username = str(
        payload.get("username", "")
    ).strip()

    password = str(
        payload.get("password", "")
    )

    if not username or not password:

        raise HTTPException(
            status_code=400,
            detail="Username/email and password are required"
        )

    user = q(
        """
        SELECT *
        FROM users
        WHERE username = %s
           OR email = %s
        LIMIT 1
        """,
        (
            username,
            username.lower()
        ),
        one=True
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid username/email or password"
        )

    # ========================================================
    # ACCOUNT LOCKOUT
    # ========================================================

    locked_until = user.get(
        "locked_until"
    )

    now = datetime.utcnow()

    if locked_until:

        # PostgreSQL timestamp may be timezone-aware
        if getattr(
            locked_until,
            "tzinfo",
            None
        ) is not None:

            current_time = datetime.now(
                timezone.utc
            )

        else:

            current_time = now

        if locked_until > current_time:

            raise HTTPException(
                status_code=429,
                detail="Account temporarily locked. Try again later."
            )

    # ========================================================
    # PASSWORD CHECK
    # ========================================================

    password_hash = user.get(
        "password_hash"
    )

    if not password_hash:

        raise HTTPException(
            status_code=500,
            detail="User password data is missing"
        )

    password_valid = check_password_hash(
        password_hash,
        password
    )

    if not password_valid:

        attempts = int(
            user.get("failed_attempts") or 0
        ) + 1

        # Five failed attempts
        if attempts >= 5:

            lock_time = datetime.utcnow() + timedelta(
                seconds=60
            )

            q(
                """
                UPDATE users
                SET
                    failed_attempts = 0,
                    locked_until = %s
                WHERE id = %s
                """,
                (
                    lock_time,
                    user["id"]
                ),
                commit=True
            )

            raise HTTPException(
                status_code=429,
                detail="Too many failed attempts. Account locked for 60 seconds."
            )

        else:

            q(
                """
                UPDATE users
                SET failed_attempts = %s
                WHERE id = %s
                """,
                (
                    attempts,
                    user["id"]
                ),
                commit=True
            )

        raise HTTPException(
            status_code=401,
            detail="Invalid username/email or password"
        )

    # ========================================================
    # SUCCESSFUL LOGIN
    # ========================================================

    q(
        """
        UPDATE users
        SET
            failed_attempts = 0,
            locked_until = NULL
        WHERE id = %s
        """,
        (
            user["id"],
        ),
        commit=True
    )

    safe_user = {
        "id": user["id"],
        "full_name": user["full_name"],
        "username": user["username"],
        "email": user["email"],
        "user_type": user["user_type"],
        "location": user["location"],
    }

    token = make_token(
        safe_user
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": _safe_user(user)
    }


# ============================================================
# CURRENT SESSION
# ============================================================

@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
    row = q(
        """
        SELECT
            id,
            full_name,
            username,
            email,
            user_type,
            location,
            avatar_filename
        FROM users
        WHERE id = %s
        """,
        (user["id"],),
        one=True
    )
    return _safe_user(row)


# ============================================================
# UPDATE PROFILE
# ============================================================

@app.put("/api/auth/profile")
def update_profile(
    payload: dict,
    user: dict = Depends(current_user)
):
    full_name = str(payload.get("full_name", "")).strip()
    username = str(payload.get("username", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    location = str(payload.get("location", user.get("location") or "")).strip()

    if not full_name or not username or not email:
        raise HTTPException(
            status_code=400,
            detail="Full name, username, and email are required"
        )

    existing = q(
        """
        SELECT id
        FROM users
        WHERE (username = %s OR email = %s)
          AND id <> %s
        LIMIT 1
        """,
        (username, email, user["id"]),
        one=True
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Username or email already registered"
        )

    row = q(
        """
        UPDATE users
        SET
            full_name = %s,
            username = %s,
            email = %s,
            location = %s
        WHERE id = %s
        RETURNING
            id,
            full_name,
            username,
            email,
            user_type,
            location,
            avatar_filename
        """,
        (full_name, username, email, location, user["id"]),
        one=True,
        commit=True
    )
    return {"message": "Profile updated", "user": _safe_user(row)}


# ============================================================
# CHANGE PASSWORD
# ============================================================

@app.post("/api/auth/change-password")
def change_password(
    payload: dict,
    user: dict = Depends(current_user)
):
    current = str(payload.get("current_password", ""))
    new_password = str(payload.get("new_password", ""))

    if not current or not new_password:
        raise HTTPException(
            status_code=400,
            detail="Current and new passwords are required"
        )
    if len(new_password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must contain at least 8 characters"
        )

    row = q(
        "SELECT password_hash FROM users WHERE id = %s",
        (user["id"],),
        one=True
    )
    if not row or not check_password_hash(row["password_hash"], current):
        raise HTTPException(
            status_code=401,
            detail="Current password is incorrect"
        )

    q(
        """
        UPDATE users
        SET password_hash = %s
        WHERE id = %s
        """,
        (generate_password_hash(new_password), user["id"]),
        commit=True
    )
    return {"message": "Password updated"}


# ============================================================
# PROFILE PHOTO
# ============================================================

@app.post("/api/auth/photo")
async def update_photo(
    image: UploadFile = File(...),
    user: dict = Depends(current_user)
):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WEBP images are allowed"
        )
    data = await image.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller"
        )
    extensions = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp"
    }
    filename = (
        "avatar_"
        + str(user["id"])
        + "_"
        + uuid.uuid4().hex
        + "."
        + extensions[image.content_type]
    )
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)

    row = q(
        """
        UPDATE users
        SET avatar_filename = %s
        WHERE id = %s
        RETURNING
            id,
            full_name,
            username,
            email,
            user_type,
            location,
            avatar_filename
        """,
        (filename, user["id"]),
        one=True,
        commit=True
    )
    return {"message": "Photo updated", "user": _safe_user(row)}


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard(user: dict = Depends(current_user)):
    identifications = q(
        "SELECT COUNT(*) AS count FROM identifications WHERE user_id = %s",
        (user["id"],),
        one=True
    )
    pests_count = q("SELECT COUNT(*) AS count FROM pests", one=True)
    posts_count = q("SELECT COUNT(*) AS count FROM community_posts", one=True)
    alerts_count = q(
        "SELECT COUNT(*) AS count FROM alerts WHERE is_active = true",
        one=True
    )
    recent = q(
        """
        SELECT
            i.id,
            i.predicted_label,
            i.confidence,
            i.image_filename,
            i.created_at,
            p.common_name,
            p.slug
        FROM identifications i
        LEFT JOIN pests p ON p.id = i.pest_id
        WHERE i.user_id = %s
        ORDER BY i.created_at DESC
        LIMIT 5
        """,
        (user["id"],)
    )
    latest_tip = q(
        """
        SELECT *
        FROM tips
        WHERE is_published = true
        ORDER BY created_at DESC
        LIMIT 1
        """,
        one=True
    )
    return {
        "identifications": int(identifications["count"] if identifications else 0),
        "pests": int(pests_count["count"] if pests_count else 0),
        "community_posts": int(posts_count["count"] if posts_count else 0),
        "active_alerts": int(alerts_count["count"] if alerts_count else 0),
        "recent_identifications": recent or [],
        "latest_tip": latest_tip,
    }


# ============================================================
# PEST INFORMATION
# ============================================================

@app.get("/api/pests")
def pests(
    _: dict = Depends(current_user)
):

    return q(
        """
        SELECT *
        FROM pests
        ORDER BY common_name
        """
    )


# ============================================================
# AGRICULTURAL TIPS
# ============================================================

@app.get("/api/tips")
def tips(
    _: dict = Depends(current_user)
):

    return q(
        """
        SELECT *
        FROM tips
        WHERE is_published = true
        ORDER BY created_at DESC
        """
    )


# ============================================================
# PEST ALERTS
# ============================================================

@app.get("/api/alerts")
def alerts(
    _: dict = Depends(current_user)
):

    return q(
        """
        SELECT *
        FROM alerts
        WHERE is_active = true
        ORDER BY created_at DESC
        """
    )


# ============================================================
# COMMUNITY POSTS - GET
# ============================================================

@app.get("/api/community")
def community(
    _: dict = Depends(current_user)
):

    return q(
        """
        SELECT
            cp.id,
            cp.title,
            cp.content,
            cp.category,
            cp.created_at,
            u.full_name,
            u.user_type
        FROM community_posts cp
        JOIN users u
            ON u.id = cp.user_id
        ORDER BY cp.created_at DESC
        """
    )


# ============================================================
# COMMUNITY POSTS - CREATE
# ============================================================

@app.post("/api/community")
def post(
    payload: dict,
    user: dict = Depends(current_user)
):

    title = str(
        payload.get("title", "")
    ).strip()

    content = str(
        payload.get("content", "")
    ).strip()

    category = str(
        payload.get("category", "Experiences")
    ).strip() or "Experiences"

    allowed_categories = {
        "Questions",
        "Experiences",
        "Success Stories",
        "Announcements",
    }
    if category not in allowed_categories:
        category = "Experiences"

    if not title or not content:

        raise HTTPException(
            status_code=400,
            detail="Title and content are required"
        )

    result = q(
        """
        INSERT INTO community_posts (
            user_id,
            title,
            content,
            category
        )
        VALUES (
            %s,
            %s,
            %s,
            %s
        )
        RETURNING *
        """,
        (
            user["id"],
            title,
            content,
            category
        ),
        one=True,
        commit=True
    )

    return result


# ============================================================
# IDENTIFICATION HISTORY
# ============================================================

@app.get("/api/history")
def history(
    user: dict = Depends(current_user)
):

    return q(
        """
        SELECT
            i.*,
            p.common_name,
            p.management
        FROM identifications i
        LEFT JOIN pests p
            ON p.id = i.pest_id
        WHERE i.user_id = %s
        ORDER BY i.created_at DESC
        """,
        (
            user["id"],
        )
    )


# ============================================================
# IMAGE IDENTIFICATION
# ============================================================

@app.post("/api/identify")
async def identify(
    image: UploadFile = File(...),
    user: dict = Depends(current_user)
):

    allowed = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if image.content_type not in allowed:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG, and WEBP images are allowed"
        )

    data = await image.read()

    if len(data) > 5 * 1024 * 1024:

        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller"
        )

    extensions = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp"
    }

    ext = extensions[
        image.content_type
    ]

    filename = (
        datetime.utcnow().strftime(
            "%Y%m%d%H%M%S%f"
        )
        + "_"
        + uuid.uuid4().hex
        + "."
        + ext
    )

    path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(
        path,
        "wb"
    ) as f:

        f.write(data)

    # ========================================================
    # CNN / YOLO PREDICTION
    # ========================================================

    try:

        prediction = predict_image(
            path
        )

    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {e}"
        )

    # ========================================================
    # FIND PEST INFORMATION
    # ========================================================

    pest = None

    if prediction.get("slug"):

        pest = q(
            """
            SELECT *
            FROM pests
            WHERE slug = %s
            LIMIT 1
            """,
            (
                prediction.get("slug"),
            ),
            one=True
        )

    # ========================================================
    # SAVE IDENTIFICATION
    # ========================================================

    detection_count = len(
        prediction.get(
            "detections",
            []
        )
    )

    record = q(
        """
        INSERT INTO identifications (
            user_id,
            pest_id,
            image_filename,
            predicted_label,
            confidence,
            detection_count,
            model_name
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        RETURNING
            id,
            created_at
        """,
        (
            user["id"],
            pest["id"] if pest else None,
            filename,
            prediction["label"],
            prediction["confidence"],
            detection_count,
            "CNN-YOLOv8"
        ),
        one=True,
        commit=True
    )

    return {
        "prediction": prediction,
        "pest": pest,
        "record": record,
        "image_url": f"/uploads/{filename}"
    }


# ============================================================
# ADMIN - USERS
# ============================================================

@app.get("/api/admin/users")
def admin_users(
    _: dict = Depends(admin_user)
):

    return q(
        """
        SELECT
            id,
            full_name,
            username,
            email,
            user_type,
            location,
            created_at
        FROM users
        ORDER BY created_at DESC
        """
    )


# ============================================================
# ADMIN - IDENTIFICATIONS
# ============================================================

@app.get("/api/admin/identifications")
def admin_identifications(
    _: dict = Depends(admin_user)
):

    return q(
        """
        SELECT
            i.*,
            u.full_name,
            p.common_name
        FROM identifications i
        JOIN users u
            ON u.id = i.user_id
        LEFT JOIN pests p
            ON p.id = i.pest_id
        ORDER BY i.created_at DESC
        LIMIT 100
        """
    )


# ============================================================
# ADMIN - TIPS
# ============================================================

@app.post("/api/admin/tips")
def admin_tip(
    payload: dict,
    _: dict = Depends(admin_user)
):

    title = str(
        payload.get("title", "")
    ).strip()

    content = str(
        payload.get("content", "")
    ).strip()

    if not title or not content:

        raise HTTPException(
            status_code=400,
            detail="Title and content required"
        )

    return q(
        """
        INSERT INTO tips (
            title,
            content
        )
        VALUES (
            %s,
            %s
        )
        ON CONFLICT(title)
        DO UPDATE SET
            content = EXCLUDED.content
        RETURNING *
        """,
        (
            title,
            content
        ),
        one=True,
        commit=True
    )


# ============================================================
# ADMIN - ALERTS
# ============================================================

@app.post("/api/admin/alerts")
def admin_alert(
    payload: dict,
    _: dict = Depends(admin_user)
):

    title = str(
        payload.get("title", "")
    ).strip()

    content = str(
        payload.get("content", "")
    ).strip()

    severity = str(
        payload.get(
            "severity",
            "INFO"
        )
    ).strip()

    if not title or not content:

        raise HTTPException(
            status_code=400,
            detail="Title and content required"
        )

    return q(
        """
        INSERT INTO alerts (
            title,
            content,
            severity
        )
        VALUES (
            %s,
            %s,
            %s
        )
        ON CONFLICT(title)
        DO UPDATE SET
            content = EXCLUDED.content,
            severity = EXCLUDED.severity,
            is_active = true
        RETURNING *
        """,
        (
            title,
            content,
            severity
        ),
        one=True,
        commit=True
    )


# ============================================================
# ADMIN - PESTS
# ============================================================

@app.post("/api/admin/pests")
def admin_pest(
    payload: dict,
    _: dict = Depends(admin_user)
):

    keys = [
        "slug",
        "common_name",
        "scientific_name",
        "description",
        "symptoms",
        "damage",
        "management"
    ]

    for key in keys:

        if not str(
            payload.get(key, "")
        ).strip():

            raise HTTPException(
                status_code=400,
                detail="All pest fields are required"
            )

    sql = """
        INSERT INTO pests (
            slug,
            common_name,
            scientific_name,
            description,
            symptoms,
            damage,
            management
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
        ON CONFLICT(slug)
        DO UPDATE SET
            common_name = EXCLUDED.common_name,
            scientific_name = EXCLUDED.scientific_name,
            description = EXCLUDED.description,
            symptoms = EXCLUDED.symptoms,
            damage = EXCLUDED.damage,
            management = EXCLUDED.management
        RETURNING *
    """

    values = tuple(
        str(
            payload[k]
        ).strip()
        for k in keys
    )

    return q(
        sql,
        values,
        one=True,
        commit=True
    )


# ============================================================
# END
# ============================================================