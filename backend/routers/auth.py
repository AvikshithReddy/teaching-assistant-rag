from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr

from auth.jwt import create_token, verify_token
from auth.password import hash_password, verify_password
from auth.users import create_user, get_user_by_email, get_user_by_id

router = APIRouter(tags=["auth"])
_bearer = HTTPBearer()


# ---------- Schemas ----------

class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str  # "professor" | "student"


class LoginBody(BaseModel):
    email: EmailStr
    password: str


# ---------- Dependency ----------

def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    try:
        payload = verify_token(creds.credentials)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = get_user_by_id(payload["sub"])
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def require_role(role: str):
    def dep(user: dict = Depends(current_user)) -> dict:
        if user["role"] != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
        return user
    return dep


# ---------- Routes ----------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterBody):
    if body.role not in ("professor", "student"):
        raise HTTPException(status_code=400, detail="role must be 'professor' or 'student'")
    if get_user_by_email(body.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    hashed = hash_password(body.password)
    user = create_user(body.email, hashed, body.name, body.role)
    token = create_token(user["id"], user["role"])
    return {"access_token": token, "token_type": "bearer", "user": {k: user[k] for k in ("id", "email", "name", "role")}}


@router.post("/login")
def login(body: LoginBody):
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(user["id"], user["role"])
    return {"access_token": token, "token_type": "bearer", "user": {k: user[k] for k in ("id", "email", "name", "role")}}


@router.get("/me")
def me(user: dict = Depends(current_user)):
    return {k: user[k] for k in ("id", "email", "name", "role")}
