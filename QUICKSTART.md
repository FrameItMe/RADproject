# Quick Start (Mac via Terminal)

คู่มือนี้อธิบายการรันโปรเจกต์ผ่าน Terminal แบบละเอียดทีละขั้น

## 1) สิ่งที่ต้องมี

1. macOS
2. Python 3 ติดตั้งแล้ว
3. อินเทอร์เน็ตสำหรับติดตั้งแพ็กเกจครั้งแรก

ตรวจสอบ Python:

python3 --version

ถ้าไม่เจอคำสั่ง ให้ติดตั้ง Python 3 ก่อน

## 2) เปิด Terminal และเข้าโฟลเดอร์โปรเจกต์

1. เปิดแอป Terminal
2. เข้าโฟลเดอร์โปรเจกต์

ตัวอย่าง:

cd /path/to/yanoey

เช็กให้ชัวร์ว่าอยู่ถูกที่:

pwd
ls

ควรเห็นไฟล์ run_mac.sh และ app.py

## 3) ตั้ง permission ให้สคริปต์ (ทำครั้งแรกครั้งเดียว)

chmod +x run_mac.sh

## 4) รันระบบ

./run_mac.sh

สคริปต์จะทำงานตามลำดับ:

1. ตรวจ Python
2. สร้าง .venv ถ้ายังไม่มี
3. ติดตั้ง dependencies จาก requirements.txt
4. เปิด Flask server
5. รอ health check แล้วเปิดเว็บให้ที่ http://127.0.0.1:5000

## 5) วิธีใช้งานเว็บหลังเปิดติด

1. เปิดเมนูมุมขวาบนเพื่อสลับหน้า
2. Page 1: Noise + Brightness
3. Page 2: Contrast + CLAHE + Undo
4. Page 3: Morphology
5. Page 4: Mammography Classification

## 6) หยุดโปรแกรม

กด Ctrl+C ในหน้าต่าง Terminal ที่กำลังรันอยู่

## 7) เปิดใหม่ครั้งถัดไป

cd /path/to/yanoey
./run_mac.sh

## 8) ถ้าเปิดไม่ติด

### อาการ: เว็บไม่ขึ้น

1. ตรวจว่า server รันอยู่หรือไม่

curl -I http://127.0.0.1:5050

2. ดู log

tail -n 80 logs/server.log

### อาการ: ติดตั้งแพ็กเกจไม่ผ่าน

1. อัปเกรด pip

python3 -m pip install --upgrade pip

2. รันใหม่

./run_mac.sh

### อาการ: พอร์ต 5050 ถูกใช้งาน

1. หา process

lsof -i :5050

2. หยุด process แล้วรันใหม่

## 9) ทางเลือกแบบดับเบิลคลิก

ถ้าไม่อยากพิมพ์คำสั่งเอง สามารถใช้ไฟล์ open_yanoey.command ได้ แต่ถ้าเจอปัญหา แนะนำให้กลับมาใช้วิธี Terminal ตามคู่มือนี้เพราะเห็น error ชัดกว่า