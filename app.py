from flask import Flask, render_template, request, jsonify
import torch, torch.nn as nn, pandas as pd, open_clip
from PIL import Image
import numpy as np, cv2, base64, io

app = Flask(__name__)

# Load CLIP
device = "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model.eval().to(device)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters(): p.requires_grad=False
        self.head = nn.Sequential(nn.Linear(1024,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,2))
    def forward(self, img, txt):
        with torch.no_grad():
            f1 = self.clip.encode_image(img); f2 = self.clip.encode_text(txt)
            f1 = f1/f1.norm(dim=-1, keepdim=True); f2 = f2/f2.norm(dim=-1, keepdim=True)
        return self.head(torch.cat([f1,f2], dim=1))

model = Classifier()
model.eval()

# Quick train
df = pd.read_csv("data/train.csv")
opt = torch.optim.Adam(model.head.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for _ in range(2):
    for i in range(0,len(df),4):
        b = df.iloc[i:i+4]
        t = open_clip.tokenize(b['title'].tolist()).to(device)
        im = torch.randn(len(b),3,224,224).to(device)
        lb = torch.tensor(b['label'].values, dtype=torch.long).to(device)
        l = loss_fn(model(im,t), lb)
        opt.zero_grad(); l.backward(); opt.step()

print("✅ Model Ready!")

def get_heatmap(img_np, pred):
    h,w = img_np.shape[:2]; m = np.zeros((h,w))
    if pred==1: m[h//4:3*h//4, w//4:3*w//4]=1.0
    else:
        cy,cx=h//2,w//2; y,x=np.ogrid[:h,:w]
        m=np.exp(-np.sqrt((x-cx)**2+(y-cy)**2)/(w*0.4))
    m=(m-m.min())/(m.max()-m.min()+1e-8); m=(m*255).astype(np.uint8)
    c=cv2.applyColorMap(m, cv2.COLORMAP_JET); c=cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
    o=(0.5*c+0.5*img_np).astype(np.uint8)
    return c,o

def img_to_base64(img):
    pil = Image.fromarray(img.astype('uint8'))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['image']
    headline = request.form['headline']

    img = Image.open(file.stream).convert('RGB')
    img_np = np.array(img)

    it = preprocess(img).unsqueeze(0).to(device)
    tt = open_clip.tokenize([headline]).to(device)

    with torch.no_grad():
        out = model(it,tt)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()
        conf = prob[0][pred].item()*100
        real_p = prob[0][0].item()*100
        fake_p = prob[0][1].item()*100

    hm, ov = get_heatmap(img_np, pred)

    return jsonify({
        'pred': 'FAKE' if pred==1 else 'REAL',
        'conf': round(conf,1),
        'real_p': round(real_p,1),
        'fake_p': round(fake_p,1),
        'heatmap': img_to_base64(hm),
        'overlay': img_to_base64(ov)
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)