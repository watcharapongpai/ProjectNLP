from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS, cross_origin  # เพิ่ม CORS
import os
import subprocess
import torch
import numpy as np
from pydub import AudioSegment
import torch.nn as nn
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from typing import Literal, List, Optional
from pydantic import BaseModel
from enum import Enum
import uuid
import whisper  # Import Whisper

app = Flask(__name__)

# อนุญาตทุกโดเมนให้เรียกใช้ API
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})
DOWNLOAD_FOLDER = os.path.join(os.getcwd(), "static")

# โหลดโมเดล Whisper
whisper_model = whisper.load_model("base")  # หรือใช้ "small", "medium", "large" ได้ตามต้องการ


# -------------------------------
# โหลดโมเดลและ word_to_index
# -------------------------------

MODEL_PATH = "models/audio_model.pth"
MAPPING_PATH = "word_mapping.json"

class SimpleAudioModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleAudioModel, self).__init__()
        self.fc1 = nn.Linear(16000, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

def load_word_to_index(mapping_path):
    with open(mapping_path, 'r') as file:
        word_to_index = json.load(file)
    return word_to_index

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

word_to_index = load_word_to_index(MAPPING_PATH)
num_classes = len(word_to_index)
model = SimpleAudioModel(num_classes=num_classes)
model = load_model(model, MODEL_PATH)

# -------------------------------
# ฟังก์ชันสำหรับดาวน์โหลด YouTube
# -------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename)

@app.route('/download', methods=['POST'])
def download():
    try:
        data = request.json
        print(f"data: {data}")  # Debugging
        youtube_url = data.get("url", None)

        if not youtube_url:
            return jsonify({"error": "No YouTube URL provided"}), 400

        print(f"กำลังดาวน์โหลด: {youtube_url}")  # Debugging

        print("📂 Flask Working Directory:", os.getcwd())  # Debugging
        if not os.path.exists(DOWNLOAD_FOLDER):
            os.makedirs(DOWNLOAD_FOLDER)

        file_name = f"downloaded_audio.mp3"
        output_filename = os.path.join(DOWNLOAD_FOLDER, file_name)
        command = [
            "yt-dlp",
            "-x",
            "--audio-format", "mp3",
            "--audio-quality", "0",       # คุณภาพสูงสุด
            "--no-part",  # ป้องกันการสร้างไฟล์ .part
            "--no-write-info-json",
            "--rm-cache-dir",  # ล้าง cache ที่ไม่จำเป็น
            "--playlist-items", "1",  # ดึงเฉพาะวิดีโอแรกของ Playlist
            "-o", output_filename,
            youtube_url
        ]
            
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            # หาชื่อไฟล์ที่ออกมาจริง

            if result.returncode == 0 :
                print(f"ดาวน์โหลดเสร็จ: {file_name}")  # Debugging
                return jsonify({"success": True, "filename": file_name })
            else:
                print("ดาวน์โหลดไม่สำเร็จ!")  # Debugging
                return jsonify({"success": False, "error": "Failed to download audio"}), 500
        except subprocess.CalledProcessError as e:
            print(f"YT-DLP Error: {str(e)}")
            return jsonify({"success": False, "error": "Failed to download audio"}), 500

    except Exception as e:
        print(f"Error in /download: {str(e)}")  # Debugging
        return jsonify({"error": str(e)}), 500
    
@app.route('/clearAudio', methods=['POST'])
@cross_origin()  # Explicitly enable CORS for this endpoint
def clearAudio():
    """ลบไฟล์ MP3 เดิมถ้าไม่มีการใช้งาน"""
    # ตรวจสอบว่าไฟล์ถูกใช้งานอยู่หรือไม่
    try:
        for file in os.listdir(DOWNLOAD_FOLDER):
            if file.endswith(".mp3"):
                file_path = os.path.join(DOWNLOAD_FOLDER, file)

                os.remove(file_path)
                print(f"ลบไฟล์เก่า: {file}")

        return jsonify({"success": "Success to remove audio"})
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการลบไฟล์ {file}: {str(e)}")
        return jsonify({"error": str(e)}), 500

# -------------------------------
# API พยากรณ์ Subtitle
# -------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        filename = os.path.join(DOWNLOAD_FOLDER, data.get("filename", None))

        print(f"Received filename: {filename}")

        if not filename or not os.path.exists(filename):
            print("File not found!")
            return jsonify({'error': 'File not found'}), 400

        predicted_subtitle = predict_subtitle(filename, word_to_index, model)
        # ใช้ Whisper เพื่อถอดเสียงเป็นข้อความ
        actual_subtitle = transcribe_audio(filename)
        # คำนวณ Accuracy
        accuracy = calculate_accuracy(predicted_subtitle, actual_subtitle)

        print(f"Predicted Subtitle")

        return jsonify({
            'predictedSubtitle': predicted_subtitle,
            'actualSubtitle': actual_subtitle,
            'accuracy': accuracy
        })

    except Exception as e:
        print(f"Error in predict(): {str(e)}")
        return jsonify({'error': str(e)}), 500

def predict_subtitle(mp3_filename, word_to_index, model):
    target_length=16000
    temperature=1.0
    audio = AudioSegment.from_mp3(mp3_filename)
    print("audio >> ", audio)
    segment_duration = len(audio) / len(word_to_index)
    predicted_words = []

    for i in range(len(word_to_index)):
        start_time = int(i * segment_duration)
        end_time = int((i + 1) * segment_duration)
        segment_audio = audio[start_time:end_time]

        audio_tensor = torch.tensor(np.array(segment_audio.get_array_of_samples()), dtype=torch.float32)

        if len(audio_tensor) < target_length:
            audio_tensor = torch.cat([audio_tensor, torch.zeros(target_length - len(audio_tensor))])
        elif len(audio_tensor) > target_length:
            audio_tensor = audio_tensor[:target_length]

        with torch.no_grad():
            output = model(audio_tensor.unsqueeze(0))
            output = output / temperature
            probabilities = torch.softmax(output, dim=1)
            predicted_label = torch.multinomial(probabilities, 1).item()

        predicted_word = list(word_to_index.keys())[predicted_label]
        predicted_words.append(predicted_word)

    return ' '.join(predicted_words)

def transcribe_audio(mp3_filename):
    """
    ใช้ OpenAI Whisper แปลงไฟล์เสียงเป็นข้อความ (Subtitle)
    """
    print(f"กำลังถอดความเสียงจากไฟล์: {mp3_filename}")
    result = whisper_model.transcribe(mp3_filename)
    subtitle_text = result["text"]
    print(f"คำบรรยายที่ได้: {subtitle_text}")
    return subtitle_text

def calculate_accuracy(predicted_text, actual_text):
    """คำนวณความแม่นยำของคำบรรยาย"""
    if not predicted_text or not actual_text:
        return 0  # หากไม่มีข้อความให้คืนค่า 0%

    # แยกคำและลบคำซ้ำ
    predicted_words = list(dict.fromkeys(predicted_text.split()))
    actual_words = list(dict.fromkeys(actual_text.split()))

    # นับคำที่ถูกต้อง
    correct_count = sum(1 for word in predicted_words if word in actual_words)

    # หลีกเลี่ยงการหารด้วยศูนย์
    accuracy = (correct_count / len(actual_words)) * 100 if actual_words else 100
    return round(accuracy, 2)


# -------------------------------
# API Recommendation
# -------------------------------

# โหลดค่าจาก .env
# ดึงค่า API Key จาก Environment Variable

@app.route('/recommendation', methods=['POST'])
def recommendation():
    """ ใช้สำหรับให้คำแนะนำจากคำบรรยายที่ได้รับ """
    try:
        data = request.json
        subtitle = data.get("subtitle", "")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not subtitle:
            return jsonify({'error': 'Subtitle is required'}), 400

        # เรียกใช้ `function_class_llm()` เพื่อจำแนกหมวดหมู่และให้คำแนะนำ
        classification_result = function_class_llm(subtitle, openai_api_key)
        print(f"classification_result : {classification_result}")
        
        return jsonify({
            'top_3_categories': str(classification_result["Top_3_Categories"]),
            'primary_category': str(classification_result["Primary_Category"]),
            'recommendation': str(classification_result["Recommendation"])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def function_class_llm(subtitle, openai_api_key):
    client = OpenAI(api_key=openai_api_key)
    openai_model  = "gpt-4o-mini"

    # Phase 1 - Top 3 Category Classification
    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": subtitle},
        ],
        temperature=0,
    )

    # Phase 1 - Parsing (Assume response is valid JSON - No Try)
    parsed_response = json.loads(completion.choices[0].message.content)

    # ดึง top_3_categories ถ้าไม่มีให้ Default เป็น Other
    top_3_categories = parsed_response.get("top_3_categories", [{"category": "Other", "confidence": 0.0}])

    # ดึง Primary Category (อันดับ 1)
    primary_category = top_3_categories[0]["category"]

    # Phase 2 - Recommendation
    completion = client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": subtitle},
        ],
        temperature=0.15,
    )

    # Phase 2 - Parsing (Assume response is valid JSON - No Try)
    recommendation_response = json.loads(completion.choices[0].message.content)

    # ดึง recommendation ถ้าไม่มีให้ Default เป็นข้อความด้านล่าง
    recommendation = recommendation_response.get("recommendation", "No recommendation available.")

    # Final Output
    output = {
        "Top_3_Categories": top_3_categories,
        "Primary_Category": primary_category,
        "Recommendation": recommendation
    }

    return output

system_prompt = """

You are a highly skilled health and wellness content categorization expert.
Your task is to classify health-related video subtitles into one of the 11 predefined categories.

You must rely **only on the provided subtitle** — do not assume additional context.
If the subtitle lacks enough detail, choose the **most appropriate category** based on the text.

---

## Categories
1. Nutrition & Diet
2. Supplements & Nutrients
3. Health Conditions & Diseases
4. Weight & Body Management
5. Fasting & Longevity
6. Mental & Emotional Health
7. Exercise & Lifestyle
8. Hormones & Metabolism
9. Natural & Alternative Remedies
10. Media & Special Topics
11. Other

---

## Category Descriptions with Example Topics and Keywords

### 1️⃣ Nutrition & Diet
- **Focus:** Whole foods, meal planning, balanced diets, keto, general eating advice.
- **Keywords:** Nutrition, Eating Tips, Keto Recipes, Sugar, ACV, Healthy Foods
- **Example Topics:**
    - "Top 10 Foods for Gut Health"
    - "How to Start the Ketogenic Diet"
    - "Best Vegetables for Fat Loss"

- ⚠️ **Nutrition & Diet vs Supplements & Nutrients**
    - If the focus is **getting nutrients from food**, choose **Nutrition & Diet**.
    - If the focus is **supplement forms, dosages, or comparing products**, choose **Supplements & Nutrients**.

---

### 2️⃣ Supplements & Nutrients
- **Focus:** Specific vitamins, minerals, supplements, dosages, forms, and comparisons.
- **Keywords:** Vitamin D, Magnesium, Zinc, Cod Liver Oil, Electrolytes
- **Example Topics:**
    - "Best Magnesium Supplement for Sleep"
    - "Benefits of Cod Liver Oil Capsules"

---

### 3️⃣ Health Conditions & Diseases
- **Focus:** Symptoms, diagnoses, treatments (both conventional and natural), diseases.
- **Keywords:** Diabetes, Cancer, Blood Pressure, Liver Health, Autoimmune
- **Example Topics:**
    - "Signs of Insulin Resistance"
    - "Natural Remedies for High Blood Pressure"

- ⚠️ **Health Conditions vs Other Categories**
    - If symptoms/diseases are the main focus, use **Health Conditions & Diseases**.
    - If content is about diet/exercise to prevent disease, use **Nutrition & Diet** or **Exercise & Lifestyle**.

---

### 4️⃣ Weight & Body Management
- **Focus:** Weight loss, fat loss, body composition, diet for body types.
- **Keywords:** Weight Loss, Belly Fat, Body Types
- **Example Topics:**
    - "Best Diet for Burning Belly Fat"
    - "How to Lose Weight Safely"

- ⚠️ **Weight & Body Management vs Exercise & Lifestyle vs Fasting**
    - If weight loss is the **primary goal**, choose **Weight & Body Management**.
    - If general fitness is the focus, use **Exercise & Lifestyle**.
    - If fasting benefits beyond weight loss are discussed, use **Fasting & Longevity**.

---

### 5️⃣ Fasting & Longevity
- **Focus:** Fasting protocols (intermittent, OMAD), autophagy, cellular repair, lifespan extension.
- **Keywords:** Intermittent Fasting, Prolonged Fasting, Autophagy, Longevity
- **Example Topics:**
    - "How Fasting Triggers Autophagy"
    - "Benefits of OMAD for Cellular Repair"

- ⚠️ **Fasting & Longevity vs Weight & Body Management**
    - Use **Fasting & Longevity** if discussing **cell repair, longevity, healthspan**.
    - Use **Weight & Body Management** if fasting is mentioned purely for weight loss.

---

### 6️⃣ Mental & Emotional Health
- **Focus:** Stress management, sleep improvement, emotional well-being, cognitive health.
- **Keywords:** Depression, Anxiety, Cognitive Health, Sleep
- **Example Topics:**
    - "Tips for Better Sleep"
    - "How Stress Affects Your Hormones"

---

### 7️⃣ Exercise & Lifestyle
- **Focus:** Exercise routines, fitness programs, healthy lifestyle tips.
- **Keywords:** Exercise, Sports, Muscle Health, Healthy Lifestyle Hacks
- **Example Topics:**
    - "Best Workouts to Burn Fat"
    - "Morning Routine for Energy"

- ⚠️ **Exercise & Lifestyle vs Weight & Body Management**
    - If **general fitness & habits** are the focus, use **Exercise & Lifestyle**.
    - If the goal is explicitly weight loss, use **Weight & Body Management**.

---

### 8️⃣ Hormones & Metabolism
- **Focus:** Hormonal balance, metabolism, insulin, cortisol, thyroid.
- **Keywords:** Insulin Resistance, Slow Metabolism, Adrenal Body Type
- **Example Topics:**
    - "Signs of Cortisol Imbalance"
    - "How Thyroid Affects Weight"

---

### 9️⃣ Natural & Alternative Remedies
- **Focus:** Herbal remedies, homeopathy, acupuncture, natural healing techniques.
- **Keywords:** Herbal Remedies, Acupressure
- **Example Topics:**
    - "Top Herbal Remedies for Inflammation"
    - "Acupressure for Stress Relief"

---

### 🔟 Media & Special Topics (Highest Priority for Promotions & Events)
- **Focus:** Promotional content, product launches, live sessions, collaborations.
- **Keywords:** Dr. Berg Live Shows, Product FAQ, Collaborations, Webinars
- **Example Topics:**
    - "Join My Live Q&A"
    - "Product Launch: New Electrolyte Powder"

---

### 1️⃣1️⃣ Other (Personal, Off-Topic, or Vague)
- **Focus:** Non-health updates, personal stories, off-topic content.
- **Example Topics:**
    - "Just got back from vacation"
    - "Let’s get started"

---

## Final Priority Rules
| Rank | Category | Use When |
|---|---|---|
| 1 | Media & Special Topics | Any promotion, event, product launch, or audience invitation |
| 2 | Other | Personal, off-topic, vague, or unrelated content |
| 3 | Exercise & Lifestyle | Fitness & healthy routines — prioritize if exercise is core |
| 4 | Fasting & Longevity | Fasting for longevity or cellular health (not weight loss) |
| 5 | Nutrition & Diet | Default for general diet advice & whole food recommendations |
| 6 | Health Conditions & Diseases | Symptoms, diseases, diagnoses, treatments |
| 7 | Supplements & Nutrients | Specific supplements, dosages, product comparisons |

---

## Output Format (Strict JSON)
{{
    "top_3_categories": [
        {{"category": "<category_name>", "confidence": <confidence_score>}},
        {{"category": "<category_name>", "confidence": <confidence_score>}},
        {{"category": "<category_name>", "confidence": <confidence_score>}}
    ],
    "recommendation": "<brief health recommendation tailored to the top category>"
}}

- Confidence ranges from 0.00 to 1.00.
- Categories must match exactly from the predefined list.

---

Ensure you strictly follow the priority rules and correctly handle overlap cases using the provided logic.
"""

class OutputOptions(str, Enum):
    Nutrition_and_Diet = "Nutrition & Diet"
    Supplements_and_Nutrients = "Supplements & Nutrients"
    Health_Conditions_and_Diseases = "Health Conditions & Diseases"
    Weight_and_Body_Management = "Weight & Body Management"
    Fasting_and_Longevity = "Fasting & Longevity"
    Mental_and_Emotional_Health = "Mental & Emotional Health"
    Exercise_and_Lifestyle = "Exercise & Lifestyle"
    Hormones_and_Metabolism = "Hormones & Metabolism"
    Natural_and_Alternative_Remedies = "Natural & Alternative Remedies"
    Media_and_Special_Topics = "Media & Special Topics"
    Other = "Other"

class RankedCategory(BaseModel):
    category: OutputOptions
    confidence: float

class Item(BaseModel):
    top_3_categories: List[RankedCategory]

class Recommendation(BaseModel):
    suggestion: str

if __name__ == '__main__':
    app.run(debug=False, port=5000)
