from openai import OpenAI
from PIL import Image
import PIL
from utils import encode_image
client = OpenAI(
    base_url="http://localhost:18903/v1",
    api_key="EMPTY",
)

messages = [
    {
        "role": "user",
        "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image('debug_echo_eval/run_20260318_192322/trajectories/479/step_0_extract_targets.jpg')}"}},
            {
                "type": "text",
                "text": """
请你定位图片中的降落伞，并输出其bounding box的坐标，按照以下格式输出： parachute: [x1, y1, x2, y2]]
                """
            }
        ]
    }
]
# messages = [
#     {
#         "role": "user",
#         "content" : "你是谁？"
#     }
# ]
response = client.chat.completions.create(messages=messages, model="qwen2.5-vl-32b", temperature=0., seed=42)

print(response.choices[0].message.content)