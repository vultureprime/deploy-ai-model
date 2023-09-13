# OpenThaiGPT 1.0.0 Beta

#### Tutorial by [![N|Solid](https://vultureprime-research-center.s3.ap-southeast-1.amazonaws.com/vulturePrimeLogo.png)](https://vultureprime.com)

#### Weight and Code by [OpenThaiGPT](https://openthaigpt.aieat.or.th/) [Colab](https://colab.research.google.com/drive/1NkmAJHItpqu34Tur9wCFc97A6JzKR8xo)

## สิ่งที่ต้องมี
- [Paperspace](https://www.paperspace.com/) Account 

## GPU Requirement 
- VRAM 24GB (A10g, A100, H100)
- Stroage 100 GB

## ขั้นตอน 
1. เข้าใช้งาน Paperspace Core Virtual Servers
2. สร้าง Virtual Machine โดยเลือกใช้งาน GPU A100-80GB 
> การสมัครใช้งานครั้งแรก จะไม่สามารถเลือก A100 ได้ จำเป็นต้องนัดหมายคุยกับทีม Support ของทาง Paperspace เพื่อขอเปิดใช้งาน A100
3. ติดตั้ง Docker 
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
4. ติดตั้ง Nvidia Container Toolkit
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && \
    sudo apt-get update
```
```
sudo apt-get install -y nvidia-container-toolkit
```
```
sudo nvidia-ctk runtime configure --runtime=docker
```
```
sudo systemctl restart docker
```
5. เริ่มต้นใช้งาน Docker 
```
sudo docker run -itd --runtime=nvidia --gpus all -v /home/paperspace:/workspace nvcr.io/nvidia/pytorch:22.08-py3
```
> ถ้ารัน  Docker แล้ว User ยังไม่เปลี่ยนเป็น Docker container สามารถใช้คำสั่ง docker exec -it {container_id} bash

> docker exec -it 4fa bash (ตัวอย่าง)
6. Verify Driver เมื่อรันคำสั่งแล้ว Cuda ควรเป็น Version 11.7
```
nvidia-smi
```

7. Upgrade pytorch จาก Version 1.x เป็น Version 2.x และติดตั้ง Library ที่จำเป็น
```
pip3 uninstall torch 
pip3 install torch torchvision torchaudio transformers accelerate bitsandbytes einops
```

8. Git clone 
```
https://github.com/vultureprime/deploy-ai-model.git && cd ./deploy-ai-model/paperspace-example/openthaigpt-1.0.0-beta-8bit-pytorch
```
9. Run 
```
mkdir llama2-openthaigpt && python3 app.py
```

## License 
MIT
