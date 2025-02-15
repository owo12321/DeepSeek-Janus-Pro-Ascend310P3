import argparse

import torch
import torch_npu
from transformers import AutoModelForCausalLM

from Janus_Pro.janus.models import MultiModalityCausalLM, VLChatProcessor

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForCausalLM
from PIL import Image
import io
import os
import time

app = FastAPI()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        default='./weight/Janus-Pro-1B',
        help="The path of model weights",
    )

    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="NPU device id",
    )

    return parser.parse_args()

args = parse_arguments()
torch.npu.set_device(args.device_id)
dtype = torch.float16
torch_npu.npu.set_compile_mode(jit_compile=False)
# specify the path to the model
model_path = args.path
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(dtype).to("npu").eval()

@torch.inference_mode()
def understanding(question):
    conversation = [
        {
            "role": "User",
            "content": f"{question}",
        },
        {"role": "Assistant", "content": ""},
    ]

    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=[], force_batchify=True
    ).to(vl_gpt.device, dtype=dtype)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@app.post("/understand_question/")
async def understand_question(
    question: str = Form(...),
):
    # 获取时间戳并格式化成下划线形式
    receive_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(f"[receive time]: {receive_time_str}")

    print(f"[question]: {question}")

    response = understanding(question)
    print(f"[response]: {response}")

    done_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(f"[done time]: {done_time_str}")

    os.mkdir(f"./logs/{receive_time_str}")
    with open(f"./logs/{receive_time_str}/log.txt", "w") as f:
        f.write(f"[receive time]: {receive_time_str}\n")
        f.write(f"[question]: {question}\n")
        f.write(f"[response]: {response}\n")
        f.write(f"[done time]: {done_time_str}\n")

    print(f"log saved at ./logs/{receive_time_str}")

    return JSONResponse({"response": response})

@torch.inference_mode()
def multimodal_understanding(image_data, question):
    conversation = [
        {
            "role": "User",
            "content": f"<image_placeholder>\n{question}",
            "images": [image_data],
        },
        {"role": "Assistant", "content": ""},
    ]

    pil_images = [Image.open(io.BytesIO(image_data))]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(vl_gpt.device, dtype=dtype)
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )
    
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

@app.post("/understand_image_and_question/")
async def understand_image_and_question(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    # 获取时间戳并格式化成下划线形式
    receive_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(f"[receive time]: {receive_time_str}")

    print(f"[filename]: {file.filename}")
    print(f"[question]: {question}")
    
    image_data = await file.read()
    response = multimodal_understanding(image_data, question)
    print(f"[response]: {response}")

    done_time_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(f"[done time]: {done_time_str}")

    os.mkdir(f"./logs/{receive_time_str}")
    with open(f"./logs/{receive_time_str}/log.txt", "w") as f:
        f.write(f"[receive time]: {receive_time_str}\n")
        f.write(f"[filename]: {file.filename}\n")
        f.write(f"[question]: {question}\n")
        f.write(f"[response]: {response}\n")
        f.write(f"[done time]: {done_time_str}\n")
    with open(f"./logs/{receive_time_str}/{file.filename}", "wb") as f:
        f.write(image_data)

    print(f"log saved at ./logs/{receive_time_str}")

    return JSONResponse({"response": response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)