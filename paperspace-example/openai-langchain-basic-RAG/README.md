# Openai Langchain Basic RAG

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

## สิ่งที่ต้องมี
- [Paperspace](https://www.paperspace.com/) Account 
- [Open AI](https://openai.com/) API Key

## ขั้นตอน 
1. เข้าใช้งาน Paperspace Core Virtual Servers
2. สร้าง Virtual Machine โดยเลือกใช้งาน CPU 
3. Git clone 
```
https://github.com/vultureprime/deploy-ai-model.git && cd ./deploy-ai-model/paperspace-example/openai-langchain-basic-RAG
```
4. สร้างไฟล์ .env สำหรับเก็บ OpenAI key 
```
echo >> .env OPENAI_API_KEY = 'xxxxxx'
```
5. Run 
```
uvicorn app:app --reload --host 0.0.0.0 --port 3000
```

## License 
MIT
