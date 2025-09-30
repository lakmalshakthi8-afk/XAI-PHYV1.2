from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = FastAPI()

# Enable CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development. In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "XAI-Phy API is running", "status": "ok"}

@app.get("/analyze")
async def analyze_text(text: str):
    try:
        # Generate visualizations
        attention_map = generate_mock_attention_map()
        attention_gif = generate_mock_attention_gif()
        
        # Analyze text for integrity issues
        hallucinations = []
        contradictions = []
        logical_consistency = []
        
        # Mock analysis based on text content
        if "moon" in text.lower():
            hallucinations.append({
                "text": "moon is made of green cheese",
                "confidence": 0.89
            })
        
        if "paris" in text.lower() and "rome" in text.lower():
            contradictions.append({
                "tokens": ["Paris", "Rome"],
                "score": 0.89
            })
        
        if any(word in text.lower() for word in ["miles", "speed", "mph"]):
            logical_consistency.append({
                "text": "100 miles in 2 hours = 50 mph",
                "is_consistent": True
            })
        
        return {
            "status": "success",
            "attention_map": attention_map,
            "attention_gif": attention_gif,
            "integrity_report": {
                "hallucinations": hallucinations,
                "contradictions": contradictions,
                "logical_consistency": logical_consistency
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_mock_attention_map():
    try:
        # Create a more interesting heatmap
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(Z, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Attention Score')
        plt.title('Attention Map')
        plt.axis('off')
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating attention map: {e}")
        raise

def generate_mock_attention_gif():
    try:
        # Create a series of more interesting frames
        size = (200, 200)
        frames = []
        
        # Generate 10 frames with moving patterns
        for i in range(10):
            # Create gradient pattern
            array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
            for x in range(size[0]):
                for y in range(size[1]):
                    r = int(255 * (np.sin(x/10 + i/3) + 1) / 2)
                    g = int(255 * (np.cos(y/10 - i/3) + 1) / 2)
                    b = int(255 * (np.sin((x+y)/20 + i/3) + 1) / 2)
                    array[x, y] = [r, g, b]
            
            img = Image.fromarray(array)
            frames.append(img)
        
        # Save to base64
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format='GIF',
            append_images=frames[1:],
            save_all=True,
            duration=100,  # Faster animation
            loop=0
        )
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error generating attention gif: {e}")
        raise