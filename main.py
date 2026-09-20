from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import urllib.request
import numpy as np
from mtcnn import MTCNN

# KinetikStudio API Başlatılıyor
app = FastAPI(title="KinetikStudio Backend")
detector = MTCNN()

class VideoRequest(BaseModel):
    image_url: str

@app.get("/")
def home():
    return {"status": "Motor Dairesi 7/24 Aktif", "version": "1.0"}

@app.post("/process-image")
def process_image(req: VideoRequest):
    try:
        # 1. Supabase'den gelen müşteri fotoğrafını hafızaya al
        req_url = urllib.request.urlopen(req.image_url)
        arr = np.asarray(bytearray(req_url.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)

        # 2. MTCNN ile analiz et
        yuzler = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not yuzler:
            return {"status": "hata", "mesaj": "Fotoğrafta yüz algılanamadı."}

        # 3. Milimetrik Izgarayı Çiz
        for yuz in yuzler:
            x, y, w, h = yuz['box']
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            step_x, step_y = w // 4, h // 4
            for i in range(1, 4):
                cv2.line(img, (x + i*step_x, y), (x + i*step_x, y+h), (0, 255, 255), 1)
                cv2.line(img, (x, y + i*step_y), (x+w, y + i*step_y), (0, 255, 255), 1)
            cv2.circle(img, (x + w//2, y + h//2), 5, (0, 0, 255), -1)

        # 4. (Sonraki aşamada bu fotoğraf Supabase'e geri yüklenecek ve Seedance'e gönderilecek)
        return {"status": "basarili", "mesaj": "Izgara eklendi, Seedance 2.0 API icin hazir."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
      
