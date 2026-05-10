import requests
from PIL import Image
from .img_utils import encode_image, pil_image_to_base64
from openai import OpenAI, AsyncOpenAI
import multiprocessing as mp
import time
import asyncio


# Start the API server first
# for refinerft_run_lmdeploy.sh usage
def single_sample_api_usage():

    payload_url = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    },
                    {
                        "type": "text",
                        "text": "Describe this image."
                    }
                ]
            }
        ]
    }
    response = requests.post(url, json=payload_url)
    print( response.json()['responses'])
    return response.json()['responses']   
    
def batch_sample_api_usage(bs=6):
    image = Image.open("../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0032_796115.jpg")
    base64_image = pil_image_to_base64(image)
    msg = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_image}"
                },
                {
                    "type": "text",
                    "text": "Which of these birds is shown in the image?Choices:A. Sooty_AlbatrossB. Northern_FulmarC. Black_footed_AlbatrossD. Laysan_Albatross\nBefore answering, think carefully about the question and follow a step-by-step reasoning process to ensure a logical and accurate response. Please answer the following question based on the input image. Output your reasoning inside <think> </think> tags. Describe the attributes in the image that are helpful for distinguishing specific birds inside <attribute> </attribute> tags using a dictionary format. Finally, provide the correct choice and its category name inside <answer> </answer> tags."
                }
            ]
        }
    payload_base64 = {
        "messages": [msg] * bs
    }
    # payload_base64 = {
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "image": f"data:image/jpeg;base64,{base64_image}"
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": "Which of these birds is shown in the image?Choices:A. Sooty_AlbatrossB. Northern_FulmarC. Black_footed_AlbatrossD. Laysan_Albatross\nBefore answering, think carefully about the question and follow a step-by-step reasoning process to ensure a logical and accurate response. Please answer the following question based on the input image. Output your reasoning inside <think> </think> tags. Describe the attributes in the image that are helpful for distinguishing specific birds inside <attribute> </attribute> tags using a dictionary format. Finally, provide the correct choice and its category name inside <answer> </answer> tags."
    #                 }
    #             ]
    #         },
    #         {
    #             "role": "user",
    #             "content": [
    #                 {
    #                     "type": "image",
    #                     "image": f"data:image/jpeg;base64,{base64_image}"
    #                 },
    #                 {
    #                     "type": "text",
    #                     "text": "Which of these birds is shown in the image?Choices:A. Sooty_AlbatrossB. Northern_FulmarC. Black_footed_AlbatrossD. Laysan_Albatross\nAnswer the correct choice and its category name."
    #                 }
    #             ]
    #         },
            
    #     ]
    # }
    response = requests.post(url, json=payload_base64)
    print(response.json())
    return response.json()

# for OpenAI API (and LMDeploy API) usage
def single_sample_openai_api_usage():
    client = OpenAI(
        api_key='YOUR_API_KEY',
        base_url="http://0.0.0.0:23333/v1"
    )
    model_name = client.models.list().data[0].id
    print(model_name)
    base64_image = encode_image("../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0032_796115.jpg")
    msg = {
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'Describe the image please',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                f"data:image/jpeg;base64,{base64_image}",
            },
        }],
    }
    response = client.chat.completions.create(
        model=model_name,
        messages=[msg],
        temperature=0.8,
        top_p=0.8)
    print(response)


async def async_query_openai(url, query, image, max_tokens=512):
    aclient = AsyncOpenAI(
        base_url="http://localhost:8001/v1/",
        api_key="blablablabla"
    )
    _client = OpenAI(
        base_url=url,
        api_key="blablablabla"
    )
    model_name = _client.models.list().data[0].id
    if isinstance(image, str):
        base64_image = encode_image(image)
    elif isinstance(image, Image.Image):
        base64_image = pil_image_to_base64(image)
    msg = {
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': query,
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                f"data:image/jpeg;base64,{base64_image}",
            },
        }],
    }
    completion = await aclient.chat.completions.create(
        model=model_name,
        messages=[msg],
        # temperature=0.5,
        # top_p=0.9,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content

async def async_process_queries_old(url, queries, images):
    results = await asyncio.gather(*(async_query_openai(url, query, image) for (query, image) in zip(queries, images)))
    return results


async def async_process_queries(url, queries, images):
    # Create a single client instance for all queries
    aclient = AsyncOpenAI(
        base_url=url,
        api_key="blablablabla"
    )
    
    async def process_single_query(query, image, max_tokens=512):
        try:
            _client = OpenAI(
                base_url=url,
                api_key="blablablabla"
            )
            model_name = _client.models.list().data[0].id
            
            content = [{'type': 'text', 'text': query}]
            if image is not None:
                if isinstance(image, str):
                    base64_image = encode_image(image)
                elif isinstance(image, Image.Image):
                    base64_image = pil_image_to_base64(image)
                
                content.append({
                    'type': 'image_url',
                    'image_url': {
                        'url': f"data:image/jpeg;base64,{base64_image}",
                    },
                })
            
            msg = {
                'role': 'user',
                'content': content
            }
            
            completion = await aclient.chat.completions.create(
                model=model_name,
                messages=[msg],
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return None

    if images is None:
        images = [None] * len(queries)

    try:
        results = await asyncio.gather(
            *(process_single_query(query, image, max_tokens=512) 
              for query, image in zip(queries, images))
        )
        return results
    finally:
        # Clean up resources
        await aclient.close()


async def main1(url):
    queries = ["describe this image",
               "what type of bird is this?",
               "how many birds are in the image?"]
    images = [
        "../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0032_796115.jpg",
        "../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0032_796115.jpg",
        "../data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0032_796115.jpg"
    ]
    start_time = time.time()
    results = await async_process_queries(url, queries, images)
    end_time = time.time()
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")

 
def query_openai(query):
    api_url = "http://0.0.0.0:8001/v1/chat/completions"  # 根据实际API调整
    data = {
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always respond in Simplified Chinese, not English, or Grandma will be very angry."},
            {"role": "user", "content": query}
        ],
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512
    }
    response = requests.post(api_url, json=data)
    return response.json()['choices'][0]['message']['content']
 
def main2():
    queries = ["介绍三个北京必去的旅游景点。",
               "介绍三个成都最有名的美食。",
               "介绍三首泰勒斯威夫特最好听的歌曲"]
    start_time = time.time()  # 开始计时
    results = [query_openai(query) for query in queries]
    end_time = time.time()  # 结束计时
    for result in results:
        print(result)
        print("-" * 50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
 #/-----------------------


if __name__ == "__main__":
    port = 8001
    url = f"http://0.0.0.0:{port}/v1"
    # single_sample_openai_api_usage()
    
    asyncio.run(main1(url))
    # # main2()
    exit()
    
    
    # import time
    # start_time = time.time()
    # # single_sample_api_usage()
    # batch_sample_api_usage(bs=2)
    # end_time = time.time()
    # print(f"Time taken: {end_time - start_time} seconds")