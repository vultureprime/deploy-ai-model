# Text-to-SQL OpenAI

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

## บทความ
- [วิธีการสร้าง Text-to-SQL เชื่อมต่อ Athena [พร้อม Code ตัวอย่าง]](https://www.vultureprime.com/how-to/how-to-build-text-to-sql-with-llamaindex-and-athena)

## สิ่งที่ต้องมี
- [AWS](https://aws.amazon.com/) Account 
- [Open AI](https://openai.com/) API Key

## สิ่งที่ต้องเตรียม
- OPENAI_API_KEY
- AWS_ACCESS_KEY
- AWS_SECRET_KEY  
- S3_STAGING_DIR (S3 สำหรับเก็บ Result)
- AWS_REGION
- DATABASE (Athena Database)
- WORKGROUP (Athena Workgroup)
- TABLE (Athena Database Table)

## ขั้นตอน 
1. เข้าใช้งาน AWS EC2 
2. สร้าง Virtual Machine โดยเลือกเป็น t3.medium (CPU)

3. Git clone 
```
https://github.com/vultureprime/deploy-ai-model.git && cd ./deploy-ai-model/aws-example/text-to-sql-openai-athena
```

4. Install dependencies
```
pip install -r requirements.txt
```

5. Run
```
uvicorn app:app --reload --host 0.0.0.0 --port 3000
```

## License 
MIT
