import requests

# 将ip.of.your.server改成服务端ip地址
ip_of_your_server = "ip.of.your.server"
# image_path为客户端图片地址, 若不使用图片, 将image_path设置为空字符串""
image_path = "./images/deepseek_logo.png"   # ""
# 输入字符串
question = "用中文描述这张图里是什么？"



understand_image_and_question_url = f"http://{ip_of_your_server}:8000/understand_image_and_question/"
understand_question_url = f"http://{ip_of_your_server}:8000/understand_question/"

# Function to call the image understanding endpoint
def understand_image_and_question(image_path, question):
    if image_path != "":
        data = {
            'question': "<image_placeholder>\n" + question,
        }
        files = {'file': open(image_path, 'rb')}
        response = requests.post(understand_image_and_question_url, files=files, data=data)
    else:
        data = {
            'question': question,
        }
        response = requests.post(understand_question_url, data=data)
    response_data = response.json()
    print("[Response]: ", response_data['response'])

# Example usage
if __name__ == "__main__":
    # Call the image understanding API
    understand_image_and_question(image_path, question)