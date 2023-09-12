# OpenThaiGPT 1.0.0 Beta

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

#### Weight and Code by [OpenThaiGPT](https://openthaigpt.aieat.or.th/) [Colab](https://colab.research.google.com/drive/1NkmAJHItpqu34Tur9wCFc97A6JzKR8xo)

## สิ่งที่ต้องมี
1. [Beam](https://beam.cloud) Account 
## ขั้นตอน 
1. ติดตั้ง Beam CLI และ Python SDK
```sh 
curl https://raw.githubusercontent.com/slai-labs/get-beam/main/get-beam.sh -sSfL | sh
```
```
pip install beam-sdk
```
2. Config Beam Account เข้ากับ CLI
```sh
beam configure --clientId={Client id ของ Account ตัวเอง} --clientSecret={Secret ของ Account ตัวเอง}
```
```sh
beam configure --clientId=xxxxxxxxe8dc7b4f18exxxxxxxxxx --clientSecret=xxxxxxxxx847cxxxxxxxxxxxxx
```
3. ดาวน์โหลดตัวอย่างโค้ดจาก Repo นี้
```
git clone https://github.com/vultureprime/awesome-beam-cloud.git
```
4. เข้าไปอย่างโฟลเดอร์ที่ต้องการ
```
cd deploy-ai-model/beam-cloud-exmaple/openthaigpt-1.0.0-beta-pytorch
```
5. Deploy ขึ้นไปที่ Beam 
```
beam deploy app.py
```
6. ทดสอบ Request  โดยสามารถดู Secert ได้จาก Call API
```
curl -X POST \
--compressed 'https://apps.beam.cloud/{URL_END_POINT}' \
-H 'Accept: */*' \
-H 'Accept-Encoding: gzip, deflate' \
-H 'Authorization: Basic {SECERT ของตัวเอง}' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-d '{"prompt" : "วิธิลดความอ้วนอย่างไร"}'
```

```
curl -X POST \
--compressed 'https://apps.beam.cloud/XXXX' \
-H 'Accept: */*' \
-H 'Accept-Encoding: gzip, deflate' \
-H 'Authorization: Basic XXXXXXiNzFiYjE4NDM5Yjk6MGM5N2I4NDdjZWE4ODI3YjXXXXXXXX' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-d '{"prompt" : "วิธิลดความอ้วนอย่างไร"}'
```

![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/Screenshot+2566-09-12+at+18.19.23.png)

## License 
MIT