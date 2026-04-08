from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from api.routes.predict_routes import router as predict_router
import os

# 1. Crear la instancia PRIMERO
app = FastAPI(
    title="IDC-Net Ensemble API",
    description="API para detección de cáncer de mama mediante ensamble de modelos",
    version="1.0.4"
)

# 2. Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")

# 3. Incluir las rutas de predicción (/api/analizar)
app.include_router(predict_router, prefix="/api")

@app.get("/")
def home():
    # Obtiene la ruta exacta de la carpeta donde está este main.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construye la ruta hacia frontend/index.html de forma segura
    ruta_html = os.path.join(BASE_DIR, "frontend", "index.html")
    
    if os.path.exists(ruta_html):
        return FileResponse(ruta_html)
    return {"mensaje": f"API lista, pero no se encontró index.html en: {ruta_html}"}