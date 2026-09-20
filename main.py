from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import urllib.request
import numpy as np

# KinetikStudio Hafif API Başlatılıyor
app = FastAPI(title="KinetikStudio Backend")

# OpenCV'nin hazır ve hafif yüz tanıma motoru
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoRequest(BaseModel):
    image_url: str

@app.get("/")
def home():
    return {"status": "Motor Dairesi 7/24 Aktif (Hafif Sürüm)", "version": "1.1"}

@app.post("/process-image")
def process_image(req_data: VideoRequest):
    try:
        # 1. Fotoğrafı bir web tarayıcısı gibi indir (403 engelini aşmak için)
        req = urllib.request.Request(
            req_data.image_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        req_url = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(req_url.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        gri_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Hafif Motor ile analiz et
        yuzler = face_cascade.detectMultiScale(gri_img, 1.1, 4)
        if len(yuzler) == 0:
            return {"status": "hata", "mesaj": "Fotoğrafta yüz algılanamadı."}

        # 3. Milimetrik Izgarayı Çiz
        for (x, y, w, h) in yuzler:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            step_x, step_y = w // 4, h // 4
            for i in range(1, 4):
                cv2.line(img, (x + i*step_x, y), (x + i*step_x, y+h), (0, 255, 255), 1)
                cv2.line(img, (x, y + i*step_y), (x+w, y + i*step_y), (0, 255, 255), 1)
            cv2.circle(img, (x + w//2, y + h//2), 5, (0, 0, 255), -1)

        return {"status": "basarili", "mesaj": "Izgara eklendi, Seedance 2.0 API icin hazir."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
