from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import gradio as gr

# Modelni yuklash
model = YOLO('best.pt')

# Ranglar
class_colors = {
    0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255),
    3: (255, 255, 0), 4: (255, 0, 255), 5: (0, 255, 255),
    6: (128, 0, 128), 7: (255, 165, 0), 8: (0, 128, 255),
    9: (255, 20, 147),
}

# Funksiya: Gradio uchun
def predict(image):
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = model(frame, conf=0.3, iou=0.3)

    class_data = defaultdict(list)
    annotated_frame = frame.copy()
    
    # Rasm o'lchamlari
    height, width = frame.shape[:2]
    object_found = False  # obyekt topildimi yoki yo‘qmi

    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                class_data[class_id].append(confidence)
                color = class_colors.get(class_id, (255, 255, 255))
                
                # ✅ Box chizish
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 5)
                object_found = True

    if not object_found:
        # ⚠️ Hech narsa topilmasa — ekran markaziga yozuv chiqaramiz
        msg = "Ma'lumot topilmadi!"
        font_scale = 5.0
        font_thickness = 10
        (text_w, text_h), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        center_x = width // 2 - text_w // 2
        center_y = height // 2 + text_h // 2
        cv2.putText(annotated_frame, msg, (center_x, center_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness)
    else:
        # Statistikani pastki chap burchakka joylashtirish
        stat_y_start = height - 50  
        
        for class_id, confidences in class_data.items():
            class_name = model.names.get(class_id, f"Class_{class_id}")
            avg_confidence = np.mean(confidences)
            avg_percent = int(avg_confidence * 100)
            detection_count = len(confidences)
            diagnosis_text = f"{class_name}: {avg_percent}% ({detection_count} ta)"
            
            stat_font_scale = 1.6
            stat_font_thickness = 4
            
            (stat_text_width, stat_text_height), _ = cv2.getTextSize(
                diagnosis_text, cv2.FONT_HERSHEY_SIMPLEX, stat_font_scale, stat_font_thickness
            )
            
            color = class_colors.get(class_id, (0, 255, 0))
            
            cv2.rectangle(annotated_frame,
                         (5, stat_y_start - stat_text_height - 10),
                         (stat_text_width + 15, stat_y_start + 10),
                         color, -1)
            
            cv2.putText(annotated_frame, diagnosis_text,
                       (10, stat_y_start), cv2.FONT_HERSHEY_SIMPLEX, stat_font_scale,
                       (255, 255, 255), stat_font_thickness)
            
            stat_y_start -= (stat_text_height + 25)
            
            if stat_y_start < stat_text_height:
                break

    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    return annotated_frame

# Gradio interfeysi
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="🦷 Dental X-ray rasmini yuklang"),
    outputs=gr.Image(type="numpy", label="🔍 Tahlil natijasi"),
    title="🦷 Dental YOLO Detektor",
    description="""
🦷 **Dental X-ray tahlil dasturi**

Bu dastur yordamida siz tish rentgen (X-ray) rasmlarini yuklab, tish va atrofdagi muhim obyektlarni ko‘rishingiz mumkin.  

📌 **Qanday ishlatish kerak:**
1. "🦷 Dental X-ray rasmini yuklang" tugmasi orqali rasm tanlang yoki yuklang.  
2. Dastur rasmni avtomatik tahlil qiladi.  
3. Natija rasmda **rangli qutilar** orqali ko‘rsatiladi.  
4. Pastki chap burchakda esa har bir obyekt nomi, aniqlik foizi va nechta obyekt topilgani ko‘rsatiladi.  
5. Agar obyekt topilmasa — ekranda **“Ma'lumot topilmadi”** degan xabar chiqadi.  

ℹ️ **Eslatma:**
- Qutilar turli **ranglarda** bo‘ladi, pastdagi yozuv ham xuddi shu rangda chiqadi.  
- Aniqlik foizi (masalan, 85%) – modelning qanchalik ishonchli topganini bildiradi.  
- Bu dastur faqat yordamchi vosita, yakuniy tashxisni faqat shifokor qo‘yadi.  

⚠️ **Diqqat!** Ushbu dastur **Beta rejimda ishlamoqda**. Natijalar ba’zida noto‘g‘ri bo‘lishi mumkin.
""",
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

# Ishga tushirish
if __name__ == "__main__":
    demo.launch(share=False, inline=True)
