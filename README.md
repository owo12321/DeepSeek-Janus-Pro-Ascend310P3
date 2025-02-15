# Janus-Pro-Ascend310P3

在华为昇腾310P3推理卡上使用mindie运行deepseek-Janus-Pro, 并部署服务化接口  
代码来源于https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MultiModal/Janus-Pro, 这份代码只适配了推理脚本inference.py, 未适配服务化推理场景.  
本仓库代码基于Janus-Pro/demo/fastapi_app.py和fastapi_client.py修改, 适配昇腾npu环境, 部署推理服务  

## 0. 环境
在以下环境亲测跑通
### 硬件
cpu: i5-10400  
内存: 32G  
npu: Atlas 300I Pro (Ascend310P3, 24G显存)  

### 系统
系统: Ubuntu 20.04.6 LTS  
内核: 5.4.0-26-generic  

### 软件
python: 3.10.16  
torch: 2.3.1+cpu  
  
cann版本: 8.0.RC3.alpha003  
driver: 24.1.rc2  
firmware: 7.3.0.1.231  
toolkit: 8.0.RC3.alpha003  
kernels: 8.0.RC3.alpha003  
mindie: 1.0.RC2  
torch_npu: 2.3.1  

环境配置过程自行参考cann官网  

## 1. 准备代码和权重
### 1.1 准备mindie Janus-Pro代码

```bash
# 在项目根目录下
git clone https://gitee.com/ascend/ModelZoo-PyTorch.git
mv ModelZoo-PyTorch/MindIE/MultiModal/Janus-Pro/* ./Janus_Pro
rm -rf ModelZoo-PyTorch

# 安装依赖
cd Janus_pro
pip install -e .
cd ..
```

### 1.2 准备hugging face deepseek-ai/Janus-Pro-1B权重
```bash
# 安装huggingface_hub: pip install huggingface_hub
huggingface-cli download deepseek-ai/Janus-Pro-1B --local-dir ./weight/Janus-Pro-1B
```

## 2. 启动
### 2.1 启动服务端
```bash
python fastapi_app.py --path='weight/Janus-Pro-1B' --device_id=0
```

### 2.2 发送请求
修改fastapi_client.py中的understand_image_and_question_url, understand_question_url, image_path, question字段, 然后执行:  
```bash
python fastapi_client.py
```

## 3. 参考
https://huggingface.co/deepseek-ai/Janus-Pro-1B  
https://www.hiascend.com/software/modelzoo/models/detail/ffe1a0f4e8ba43aeb989251a3f0308e9  
https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MultiModal/Janus-Pro  
