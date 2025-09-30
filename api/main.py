from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "XAI-Phy API is running"}

@app.post("/analyze")
async def analyze_text(text: str):
    # Mock attention visualization (replace with your actual model)
    attention_map = generate_mock_attention_map()
    attention_gif = generate_mock_attention_gif()
    
    return {
        "attention_map": attention_map,
        "attention_gif": attention_gif,
        "integrity_report": {
            "hallucinations": [
                {
                    "text": "moon is made of green cheese",
                    "confidence": 0.89
                }
            ],
            "contradictions": [
                {
                    "tokens": ["Paris", "Rome"],
                    "score": 0.89
                }
            ],
            "logical_consistency": [
                {
                    "text": "100 miles in 2 hours = 50 mph",
                    "is_consistent": True
                }
            ]
        }
    }

def generate_mock_attention_map():
    # Create a simple heatmap
    data = np.random.rand(10, 10)
    plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='viridis')
    plt.colorbar()
    plt.title('Attention Map')
    
    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_mock_attention_gif():
    # Create a series of frames
    frames = []
    for i in range(5):
        img = Image.new('RGB', (100, 100), color='black')
        frames.append(img)
    
    # Save to base64
    buf = io.BytesIO()
    frames[0].save(
        buf,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=500,
        loop=0
    )
    return base64.b64encode(buf.getvalue()).decode('utf-8')