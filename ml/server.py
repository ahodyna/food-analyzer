from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from predict_pytorch import FoodPredictor

app = FastAPI()

# Ініціалізуємо predictor один раз при старті сервера
try:
    predictor = FoodPredictor(model_path='./models')
    print("Food predictor initialized successfully!")
except Exception as e:
    print(f"Error initializing predictor: {e}")
    predictor = None

@app.get("/")
async def root():
    return {"message": "Food Recognition API is running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None
    }

@app.post("/predict")
async def predict_food(file: UploadFile = File(...), top_k: int = 5):
    """
    Predict food type from uploaded image
    """
    # Перевіряємо чи predictor завантажився
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Перевіряємо тип файлу
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    # Перевіряємо top_k параметр
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    try:
        # Читаємо bytes з файлу
        image_bytes = await file.read()
        
        # Перевіряємо чи файл не порожній
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Викликаємо модель для передбачення
        result = predictor.predict_from_bytes(image_bytes, top_k=top_k)
        
        # Додаємо додаткову інформацію до відповіді
        result['file_info'] = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size_bytes': len(image_bytes)
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        # Перекидаємо HTTP exceptions як є
        raise
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict-batch")
async def predict_food_batch(files: list[UploadFile] = File(...), top_k: int = 5):
    """
    Predict food types for multiple uploaded images
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    if top_k < 1 or top_k > 10:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 10")
    
    results = []
    
    for file in files:
        try:
            # Перевіряємо тип файлу
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'File is not an image'
                })
                continue
            
            # Читаємо bytes з файлу
            image_bytes = await file.read()
            
            if len(image_bytes) == 0:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Empty file'
                })
                continue
            
            # Викликаємо модель для передбачення
            result = predictor.predict_from_bytes(image_bytes, top_k=top_k)
            
            # Додаємо інформацію про файл
            result['file_info'] = {
                'filename': file.filename,
                'content_type': file.content_type,
                'size_bytes': len(image_bytes)
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    return JSONResponse(content={'results': results})

if __name__ == "__main__":
    # For development - run without reload when executed directly
    uvicorn.run(app, host="0.0.0.0", port=5000)