# ML Notes

โฟลเดอร์นี้เก็บส่วนที่เกี่ยวกับ machine learning ทั้งหมดของโปรเจกต์ ไม่ว่าจะเป็นการเตรียมข้อมูล การฝึกโมเดล การทดสอบผล และเครื่องมือเสริมสำหรับแปลงโมเดลไปใช้ในรูปแบบอื่น

ถ้าคุณต้องการคู่มือแบบเริ่มจากศูนย์ให้ดู [README.md](../README.md) ที่โฟลเดอร์หลักก่อน เพราะไฟล์นี้จะเน้นอธิบายเฉพาะงานฝั่งโมเดล

## ไฟล์แต่ละตัวในโฟลเดอร์นี้ทำอะไร

- `prepare_mias_dataset.py`
  - อ่านข้อมูล MIAS จากโฟลเดอร์ `data/`
  - แปลงภาพจาก `.pgm` เป็น `.png`
  - จัดภาพลงโครงสร้าง `train/`, `val/`, `test/`
  - แยกคลาสเป็น `normal`, `benign`, `malignant`
  - ลบ `dataset/` เดิมแล้วสร้างใหม่ถ้ารันซ้ำ เพื่อให้ชุดข้อมูลไม่ปนของเก่า
- `train_mammogram_model.py`
  - เป็นสคริปต์ฝึกโมเดลหลักของโปรเจกต์
  - ใช้ EfficientNetB0 แบบ transfer learning
  - ฝึกโมเดลแบบ 3 คลาส
  - คำนวณ class weights ได้
  - ประเมินผลบน validation และ test set
  - สร้าง confusion matrix และ calibration threshold
  - บันทึกไฟล์ผลลัพธ์ลง `artifacts/`
- `inference_server.py`
  - เป็น Flask server แยกเดี่ยวสำหรับทดสอบการทำนาย
  - รับรูปภาพผ่าน API `/classify`
  - เหมาะกับกรณีที่อยากเช็กโมเดลเฉย ๆ โดยไม่ต้องเปิดเว็บตัวเต็ม
  - ปัจจุบันโปรเจกต์หลักใช้ `app.py` ที่โฟลเดอร์บนสุดเป็นตัวหลักมากกว่า
- `convert_to_tfjs.py`
  - ใช้แปลงโมเดล `.keras` ไปเป็น TensorFlow.js
  - มีประโยชน์ถ้าอยากเอาโมเดลไปรันใน browser โดยตรง
  - เป็นเครื่องมือเสริม ไม่ใช่ขั้นตอนบังคับของการใช้งานเว็บปกติ
- `requirements.txt`
  - รายการแพ็กเกจเฉพาะฝั่ง ML
  - ใช้เมื่ออยากติดตั้งเฉพาะเครื่องมือสำหรับเตรียมข้อมูลหรือฝึกโมเดล
- `README.md`
  - เอกสารอธิบายโฟลเดอร์ ML ฉบับนี้

## โครงสร้างข้อมูลที่ต้องมี

ก่อนฝึกโมเดล ข้อมูลควรถูกจัดไว้ในรูปแบบนี้:

```text
dataset/
  train/
    normal/
    benign/
    malignant/
  val/
    normal/
    benign/
    malignant/
  test/
    normal/
    benign/
    malignant/
```

ความหมายของแต่ละโฟลเดอร์:

- `train/` คือชุดสำหรับสอนโมเดลให้เรียนรู้
- `val/` คือชุดสำหรับตรวจสอบระหว่างฝึกว่าโมเดลเริ่ม overfit หรือยัง
- `test/` คือชุดสำหรับวัดผลสุดท้ายหลังฝึกเสร็จ
- `normal/`, `benign/`, `malignant/` คือชื่อคลาสของภาพ

ชุดข้อมูลปัจจุบันใช้ MIAS Mammography dataset ซึ่งเป็นชุดข้อมูลภาพแมมโมแกรมที่แบ่ง label ไว้แล้ว

## ขั้นตอนเตรียมข้อมูล

ถ้าคุณเพิ่งได้ไฟล์ข้อมูลมาใหม่ ให้ทำตามลำดับนี้:

### 1. ตรวจให้แน่ใจว่ามีข้อมูลดิบใน `data/`

โฟลเดอร์ `data/` ควรมีไฟล์ MIAS ต้นฉบับ เช่น:

- โฟลเดอร์ `all-mias/`
- ไฟล์ `Info.txt`
- ไฟล์ภาพ `.pgm`

### 2. รันสคริปต์เตรียม dataset

```bash
python ml/prepare_mias_dataset.py
```

สิ่งที่สคริปต์นี้ทำ:

- อ่าน `Info.txt`
- หา label ของแต่ละภาพ
- แปลง `.pgm` เป็น `.png`
- จัดลง `dataset/train`, `dataset/val`, `dataset/test`

### 3. ตรวจผลลัพธ์

หลังรันเสร็จ ควรเห็นโครงสร้าง `dataset/` ที่พร้อมใช้งาน

ถ้ารันซ้ำ สคริปต์จะลบชุด `dataset/` เดิมแล้วสร้างใหม่ เพื่อหลีกเลี่ยงปัญหาข้อมูลค้างจากรอบก่อน

## ขั้นตอนฝึกโมเดล

ถ้าคุณมี `dataset/` พร้อมแล้ว ให้รันคำสั่งนี้:

```bash
python ml/train_mammogram_model.py --data_dir dataset --out_dir artifacts
```

คำสั่งนี้จะ:

- โหลดชุดข้อมูลจาก `dataset/`
- ฝึกโมเดล EfficientNetB0
- เก็บโมเดลที่ดีที่สุดระหว่างฝึก
- ประเมินผลบน test set
- สร้าง confusion matrix บน validation set
- สร้าง calibration threshold สำหรับการทำนาย
- เขียนไฟล์ผลลัพธ์ลง `artifacts/`

ถ้าต้องการปิด class weights ระหว่างฝึก ให้ใช้:

```bash
python ml/train_mammogram_model.py --data_dir dataset --out_dir artifacts --no_class_weights
```

## ไฟล์ผลลัพธ์ที่ควรรู้จัก

- `artifacts/mammogram_classifier.keras`
  - โมเดลหลักสำหรับใช้ทำนายผลจริง
- `artifacts/best_model.keras`
  - โมเดล checkpoint ที่เก็บตอนช่วงที่ดีที่สุดของการฝึก
- `artifacts/class_map.json`
  - ตารางแปลงเลขคลาสเป็นชื่อคลาส
- `artifacts/metrics.json`
  - ผลประเมินบน test set เช่น accuracy, AUC และ loss
- `artifacts/val_confusion_matrix.json`
  - confusion matrix และรายงานผลของ validation set
- `artifacts/calibration.json`
  - threshold ที่ใช้ช่วยเลือกคลาสตอน inference

## ถ้าจะทดสอบ API แยกเดี่ยว

ถ้าอยากทดสอบโมเดลโดยไม่เปิดเว็บหลัก สามารถใช้เซิร์ฟเวอร์ใน `inference_server.py` ได้

โดยหลักการคือ:

1. เปิดเซิร์ฟเวอร์ Flask ฝั่ง ML
2. ส่งรูปไปที่ endpoint `/classify`
3. อ่านผล class และ confidence ที่ตอบกลับมา

อย่างไรก็ตาม เวอร์ชันปัจจุบันของโปรเจกต์ใช้ `app.py` ที่โฟลเดอร์บนสุดเป็นทางหลักสำหรับใช้งานจริง

## ถ้าจะส่งออกโมเดลไปใช้ใน browser

ไฟล์ `convert_to_tfjs.py` เอาไว้แปลงโมเดลไปเป็น TensorFlow.js

กรณีใช้งาน:

- ถ้าคุณอยากให้โมเดลรันใน browser แบบไม่เรียก Flask
- ถ้าต้องการทำเดโมที่โหลดโมเดลจากไฟล์ฝั่ง frontend

แต่สำหรับการใช้งานทั่วไปของโปรเจกต์นี้ ไม่จำเป็นต้องแปลงเป็น TFJS ก็ได้

## หมายเหตุสำคัญ

- ไฟล์ในโฟลเดอร์นี้ส่วนใหญ่ไม่ใช่สิ่งที่ผู้ใช้ทั่วไปต้องแตะทุกวัน
- ถ้าคุณแค่ต้องการเปิดเว็บใช้งาน ให้ใช้ `open_yanoey.command` หรือ `run_mac.sh` จากโฟลเดอร์หลัก
- ถ้าคุณต้องการ retrain โมเดล ให้เริ่มจาก `prepare_mias_dataset.py` ก่อน แล้วค่อย `train_mammogram_model.py`
