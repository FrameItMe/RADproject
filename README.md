# Yanoey Mammogram Web App

เอกสารนี้เป็นคู่มือหลักของโปรเจกต์ โดยเน้น 3 อย่าง

1. โปรเจกต์นี้ทำอะไร
2. ส่วน Machine Learning ทำอะไรบ้าง
3. Pipeline ตั้งแต่ข้อมูลดิบไปจนถึงโมเดลที่ใช้งานจริงในเว็บ

ถ้าอ่านไฟล์นี้จบ คุณควรเห็นภาพรวมระบบทั้งหมดและรู้ว่าแต่ละขั้นต้องทำอะไร

## 1) Project Overview

Yanoey Mammogram Web App คือเว็บแอปที่รวม

1. เครื่องมือประมวลผลภาพแมมโมแกรมแบบ basic image processing
2. ระบบจำแนกภาพด้วยโมเดล ML (3 คลาส: normal, benign, malignant)
3. Backend API และหน้าเว็บในเซิร์ฟเวอร์ Flask ตัวเดียว

แนวคิดหลักของโปรเจกต์นี้คือทำให้ผู้ใช้เปิดใช้งานได้ง่ายบนเครื่องเดียว (single-machine workflow) โดยไม่ต้องแยก frontend server และ backend server

## 2) What We Built

ระบบประกอบด้วย 4 หน้าหลักใน UI

1. หน้า 1: ลด noise และปรับ brightness
2. หน้า 2: ปรับ contrast และบันทึกภาพ
3. หน้า 3: ทำ morphology
4. หน้า 4: เรียก Model API เพื่อ classify ภาพ

และมี 2 แกนหลักของงาน

1. Product/Application track: เว็บใช้งานจริง, API, UX, startup scripts
2. ML track: data prep, training, calibration, hard-case audit, promotion

## 3) Repository Structure (Current)

โครงสร้างระดับบนสุดที่ควรรู้

1. app.py
2. web/
3. ml/
4. datateacher/
5. dataset/
6. artifacts/
7. experiments/
8. logs/
9. data/ (legacy raw source)
10. run_mac.sh, open_yanoey.command, open_yanoey.bat

คำอธิบายสั้น

1. app.py: Flask server หลัก (เสิร์ฟเว็บ + API ทำนาย)
2. web/: Frontend (HTML/CSS/JS)
3. ml/: สคริปต์ฝั่ง ML
4. datateacher/: แหล่งข้อมูลหลักที่ใช้เตรียม dataset ปัจจุบัน
5. dataset/: ข้อมูลที่เตรียมแล้วสำหรับ train/val/test
6. artifacts/: โมเดลและไฟล์ที่ production ใช้งานจริง
7. experiments/: งานทดลองทั้งหมด (โมเดลรอบทดลอง, dataset variant, backups, audits)
8. logs/: log runtime เช่น logs/server.log
9. data/: ข้อมูลเก่าจากรอบก่อนหน้า (ยังเก็บไว้เพื่ออ้างอิง)

## 4) End-to-End System Architecture

สถาปัตยกรรม runtime ของระบบตอนใช้งานจริง

1. Browser โหลดหน้าเว็บจาก Flask
2. ผู้ใช้อัปโหลดภาพและปรับภาพบนหน้าเว็บ
3. หน้า 4 เรียก POST /classify ไปที่ Flask
4. Flask โหลดโมเดลจาก artifacts/
5. Flask preprocess ภาพ + ทำ inference
6. Flask ส่งผล class + confidence + probabilities กลับ
7. Frontend แสดงผลผู้ใช้

ข้อดี

1. Deployment ง่าย (server เดียว)
2. Debug ง่าย (log จุดเดียว)
3. ลดปัญหา CORS/port mismatch ด้วย same-origin design

## 5) Detailed ML Pipeline

ส่วนนี้คือหัวใจของโปรเจกต์

### 5.1 Data Source and Labeling

แหล่งข้อมูลหลัก

1. datateacher/extracted/classification
2. ไฟล์ metadata หลักคือ mias_derived_info.csv

หมายเหตุสำคัญ

1. รอบ train ปัจจุบันใช้ datateacher เป็นหลัก ไม่ได้ train จาก data/
2. โฟลเดอร์ data/ เป็นแหล่งข้อมูลเก่าที่เก็บไว้เพื่ออ้างอิงย้อนหลัง

การแปลง label เป็น 3 คลาส

1. normal -> normal
2. benign/benige -> benign
3. malignant -> malignant

### 5.2 Dataset Preparation

สคริปต์หลัก

1. ml/prepare_datateacher_dataset.py

หน้าที่

1. อ่านภาพและ label จาก datateacher/extracted/classification
2. ทำ stratified split เป็น train/val/test
3. จัดไฟล์ลง class folders ให้เป็นมาตรฐานเดียวกับ training pipeline

สคริปต์เก่า (legacy)

1. ml/prepare_mias_dataset.py ยังอยู่ใน repo สำหรับงานเก่า แต่ไม่ใช่เส้นทางหลักของการ train ปัจจุบัน

ผลลัพธ์

1. dataset/train/normal|benign|malignant
2. dataset/val/normal|benign|malignant
3. dataset/test/normal|benign|malignant

### 5.3 Training Pipeline

สคริปต์หลัก

1. ml/train_mammogram_model.py

Model backbone

1. DenseNet121 (transfer learning)

Input pipeline

1. โหลดภาพด้วย image_dataset_from_directory
2. แปลงเป็น grayscale แล้ว stack กลับเป็น 3 channel
3. normalize เป็นช่วง 0..1

Data balancing

1. Balanced sampling ต่อคลาสใน train pipeline
2. Class weights คำนวณจาก distribution ของ train split
3. รองรับ minority boost สำหรับ benign/malignant ในรอบ fine-tune เฉพาะทาง

Loss and optimization

1. Focal loss เพื่อลดปัญหา class imbalance
2. Stage 1 และ Stage 2 fine-tuning
3. ReduceLROnPlateau + EarlyStopping + ModelCheckpoint

Metrics ที่ใช้ monitor

1. val_macro_f1
2. val_balanced_accuracy
3. accuracy, auc เป็นตัวสนับสนุน

### 5.4 Calibration and Decision Thresholds

หลัง train จะไม่ใช้ argmax อย่างเดียว แต่ทำ threshold calibration บน validation

ไฟล์ที่ได้

1. artifacts/calibration.json

หลักการ

1. ค้น grid ของ threshold แต่ละคลาส
2. optimize ตาม macro_f1 แล้วตามด้วย balanced_accuracy
3. เซฟ threshold ที่ดีที่สุดไปใช้ตอน inference จริง

### 5.5 Inference Pipeline in Production

เมื่อ API /classify รับภาพเข้ามา

1. preprocess_image
   1. แปลง grayscale
   2. crop black border
   3. equalize/resize ตาม pipeline inference
2. predict_with_ensemble
   1. base view
   2. bright-crop view
3. TTA (flip/brightness variants)
4. choose_class_index + calibration thresholds
5. uncertainty/hotspot gate เพื่อลด false decision บางกรณี

Output JSON

1. class
2. confidence
3. decision_source
4. probabilities (normal/benign/malignant)

### 5.6 Hard-Case Audit and Iterative Improvement

เครื่องมือสำคัญ

1. ml/audit_hard_cases.py

หน้าที่

1. สแกน train/val/test
2. หาตัวอย่าง benign/malignant ที่ทายพลาดหรือ true confidence ต่ำ
3. เขียนสรุปออกเป็น CSV และ JSON

ตำแหน่งผลลัพธ์

1. experiments/audit/hard_cases/

การใช้จริงในรอบพัฒนา

1. เลือก hard cases ที่สำคัญ (เช่น malignant->benign)
2. ทำ dataset variant ใน experiments/datasets
3. short fine-tune จากโมเดลปัจจุบัน
4. วัดผลเทียบ baseline
5. promote เฉพาะรอบที่ชนะจริง

### 5.7 Promotion Policy

หลักการ promote model

1. ไม่ promote เพราะความรู้สึก
2. ต้องชนะ baseline บน metric สำคัญ (macro_f1, balanced_accuracy และ behavior per-class)
3. backup artifacts เดิมทุกครั้งก่อน promote
4. promote เฉพาะไฟล์จำเป็นเข้า artifacts/

backup policy

1. เก็บ snapshot เดิมไว้ใน experiments/models/backups/

## 6) Current Production Artifacts

ไฟล์ที่ runtime ใช้งานจริง

1. artifacts/best_model.keras
2. artifacts/mammogram_classifier.keras
3. artifacts/class_map.json
4. artifacts/calibration.json
5. artifacts/metrics.json
6. artifacts/val_confusion_matrix.json

คำอธิบาย

1. best_model.keras: best checkpoint
2. mammogram_classifier.keras: exported serving model
3. class_map.json: index -> class name
4. calibration.json: thresholds + calibration details
5. metrics.json: test metrics ของโมเดลที่ใช้อยู่
6. val_confusion_matrix.json: validation report และ confusion matrices

## 7) Experiments and Reproducibility

เราจัดงานทดลองไว้ใน experiments/ เพื่อแยกจาก production

1. experiments/models/: ผล train แต่ละรอบ
2. experiments/datasets/: dataset variants
3. experiments/audit/: hard-case reports
4. experiments/models/backups/: backup ก่อน promote

หลักการใช้งาน

1. อย่าเขียนทับ artifacts/ โดยตรงระหว่างทดลอง
2. เทรนออก experiments/models ก่อน
3. เทียบ baseline ให้ชัด
4. ค่อย promote เข้าสู่ artifacts/

## 8) How To Run (Runtime)

### 8.1 macOS (recommended)

1. เปิด Terminal
2. เข้าโฟลเดอร์โปรเจกต์
3. รัน

bash run_mac.sh

ระบบจะ

1. เตรียม virtual environment
2. ติดตั้ง dependencies
3. เปิด Flask ที่พอร์ต 5050
4. บันทึก log ไป logs/server.log
5. เปิดเว็บอัตโนมัติ

### 8.2 macOS Double Click

1. ดับเบิลคลิก open_yanoey.command
2. ระบบจะรัน flow ใกล้เคียงกับ run_mac.sh

### 8.3 Windows Double Click

1. ดับเบิลคลิก open_yanoey.bat

## 9) How To Train

### 9.0 Prepare Dataset (Current Source)

python ml/prepare_datateacher_dataset.py

คำสั่งนี้จะสร้าง/อัปเดต dataset/ จาก datateacher/extracted/classification

### 9.1 Standard Train

python ml/train_mammogram_model.py --data_dir dataset --out_dir experiments/models/run_name

### 9.2 Fine-Tune from Existing Model (Short Round)

python ml/train_mammogram_model.py --data_dir dataset --out_dir experiments/models/run_name --init_model artifacts/best_model.keras --stage1_epochs 6 --stage2_epochs 4 --minority_boost 1.2

### 9.3 Hard-Case Audit

python ml/audit_hard_cases.py --data_dir dataset --model_path artifacts/best_model.keras --out_dir experiments/audit/hard_cases --top_k 60

## 10) ML Evaluation Criteria

เราให้ความสำคัญ metric แบบนี้

1. Macro F1: วัดความสมดุลทุกคลาส
2. Balanced Accuracy: ช่วยลดภาพลวงจาก class imbalance
3. Per-class recall/precision: ดู benign และ malignant แยกชัด
4. Confusion matrix: ดู pattern การสับสน เช่น malignant ถูกทายเป็น benign

เกณฑ์ practical ในงานนี้

1. หลีกเลี่ยง normal bias
2. ไม่ยอมให้ benign หรือ malignant collapse
3. ตัดสินใจจาก report จริง ไม่ใช่ accuracy อย่างเดียว

## 11) API Contract (Production)

### 11.1 Health Check

GET /health

response ตัวอย่าง

1. status
2. model_loaded
3. model_file

### 11.2 Classification

POST /classify

request body

1. image (base64)

response

1. class
2. confidence
3. decision_source
4. probabilities
5. hotspot (ถ้ามี)

## 12) Troubleshooting

ปัญหาพบบ่อย

1. โมเดลโหลดไม่ได้
   1. เช็กไฟล์ใน artifacts/
   2. ดู logs/server.log
2. พอร์ตชน
   1. ตอนนี้ใช้พอร์ต 5050
   2. เช็กด้วย lsof -i :5050
3. classify ไม่ตอบ
   1. เช็ก /health ก่อน
   2. เช็ก CORS/origin และ log
4. train ค้างหรือช้ามาก
   1. ลด epochs
   2. ลด stage2
   3. ตรวจ path ของ dataset

## 13) Project Scope and Limitations

สิ่งที่โปรเจกต์นี้เหมาะ

1. การเรียนรู้ pipeline ภาพแมมโมแกรมแบบ end-to-end
2. การทดลอง balancing/calibration/hard-case iteration
3. prototype เพื่อสาธิตแนวทาง ML-assisted screening

สิ่งที่โปรเจกต์นี้ยังไม่ใช่

1. ไม่ใช่ medical device
2. ไม่ใช่ระบบวินิจฉัยทางการแพทย์ที่รับรองทางคลินิก
3. ไม่ควรใช้แทนผู้เชี่ยวชาญด้านรังสีวิทยา

## 14) Suggested Next Steps

ถ้าจะพัฒนาต่อแบบ research-to-product

1. ทำ cross-validation เพิ่มความน่าเชื่อถือของ metric
2. เพิ่ม model explainability เช่น Grad-CAM
3. เพิ่ม data quality audit และ label audit pipeline
4. เก็บ experiment metadata แบบเป็นระบบ (เช่น run registry)
5. ทำ CI check สำหรับ training/evaluation reproducibility

## 15) Quick Map

ถ้าจะหาไฟล์เร็ว

1. Runtime server: app.py
2. Frontend logic: web/app.js
3. ML training: ml/train_mammogram_model.py
4. Dataset prepare (current): ml/prepare_datateacher_dataset.py
5. Hard-case audit: ml/audit_hard_cases.py
6. Production artifacts: artifacts/
7. Experiments and backups: experiments/
8. Runtime logs: logs/server.log

## 16) Recent Updates (Robust CV)

**1. Robust Image Processing Estimators (หน้า 1 & 2)**
แก้ไขบั๊กคณิตศาสตร์ของการประมวลผลภาพอัตโนมัติ เพื่อไม่ให้รวนเวลาเจอตัวหนังสือสีขาว (Text) หรือขอบฟิล์มสีดำ:
- **Auto Noise:** เปลี่ยนสมการเป็น **Median Absolute Deviation (MAD)** และตัดพิกเซลขยะทิ้ง 10% เพื่อให้คำนวณเฉพาะเม็ด Noise แท้ๆ
- **Auto Contrast:** เปลี่ยนสมการการดึงแสงเป็น **Percentile-based Normalization** (ตัดหัวท้าย 5%) เพื่อไม่ให้ขอบภาพสีดำหรือตัวหนังสือสีขาวมาดึงความสว่างจนเพี้ยน
