from fastapi import APIRouter, UploadFile, File, HTTPException
from api.controllers.predict_controllers import run_ensemble_prediction

# Creamos el router (como el que mostró tu profe en sus capturas)
router = APIRouter()

@router.post("/analizar")
async def predict(file: UploadFile = File(...)):
    # 1. Validar que sea una imagen
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen (JPG/PNG)")

    try:
        # 2. Llamar al controlador para procesar la imagen
        # Usamos .file para pasar el stream de datos directamente
        resultado = run_ensemble_prediction(file.file)
        
        # 3. Responder al Front-end
        return {
            "status": "success",
            "data": resultado
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")