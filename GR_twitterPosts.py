"""
UPDATED JANUARY 21 - 2025
To fix the HF API TOKEN STUFF___DONE
"""

import datetime, random, string
import gradio as gr
#from openai import OpenAI
from gradio_client import Client
from PIL import Image
from rich.console import Console
import os
from huggingface_hub import InferenceClient


console = Console(width=80)
theme=gr.themes.Default(primary_hue="blue", secondary_hue="pink",
                        font=[gr.themes.GoogleFont("Oxanium"), "Arial", "sans-serif"]) 

def checkHFT(hf_token):
    if 'hf_' in hf_token:
        return gr.Row(visible=True),gr.Row(visible=True),gr.Row(visible=True),gr.Row(visible=True),"✅HF TOKEN detected"
   
    else:
        gr.Warning("⚠️ You don't have a Hugging Face Token set")
        return gr.Row(visible=False),gr.Row(visible=False),gr.Row(visible=False),gr.Row(visible=False), "⚠️ You don't have a Hugging Face Token set"  
    

def writehistory(filename,text):
    """
    save a string into a logfile with python file operations
    filename -> str pathfile/filename
    text -> str, the text to be written in the file
    """
    with open(f'{filename}', 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

def genRANstring(n):
    """
    n = int number of char to randomize
    Return -> str, the filename with n random alphanumeric charachters
    """
    N = n
    res = ''.join(random.choices(string.ascii_uppercase +
                                string.digits, k=N))
    return f'Logfile_{res}.txt'

LOGFILENAME = genRANstring(5)

################## STABLE DIFFUSION PROMPT ##############################
def createSDPrompt(token,headers):
    #bruteText = bruteText.replace('\n\n','\n')
    SD_prompt = f'''Create a prompt for Stable Diffusion based on the information below. Return only the prompt.\n---\n{headers}\n\nPROMPT:'''
    client = InferenceClient(token=token)
    messages = [{"role": "user", "content": SD_prompt}]
    completion = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=messages,
        max_tokens=500
    )
    print(completion.choices[0].message.content)
    ImageGEN_prompt = completion.choices[0].message.content
    return ImageGEN_prompt

############### CREATE IMAGE ##########################
def CreateImage(token,ImageGEN_prompt):
    from gradio_client import Client
    from gradio_client import handle_file
    from PIL import Image
    client = Client("stabilityai/stable-diffusion-3.5-large",hf_token=token)
    result = client.predict(
            prompt=ImageGEN_prompt,
        negative_prompt='blur',
            seed=0,
            randomize_seed=True,
            width=1360,
            height=768,
            guidance_scale=4.5,
            num_inference_steps=30,
            api_name="/infer"
    )
    ############ SAVE IMAGE ##########################
    from gradio_client import handle_file
    temp = result[0]
    from PIL import Image
    image = Image.open(temp)
    imagename = datetime.datetime.strftime(datetime.datetime.now(),'IMage_%Y-%m-%d_%H-%M-%S.png')
    image.save(imagename)
    print(f'Image saved as {imagename}...')
    return image, imagename

def openDIR():
    import os
    current_directory = os.getcwd()
    print("Current Directory:", current_directory)
    os.system(f'start explorer "{current_directory}"')

############# TWEET GENERATION #########################
def createTweets(token,bruteText):   
    Tweet_prompt = f"Read the following newsletter. rewrite it into 3 twitter posts in English, in progression.\n---\n{bruteText}"
    from rich.console import Console
    console = Console(width=80)
    # using https://huggingface.co/spaces/eswardivi/phi-4
    client = Client("eswardivi/phi-4",hf_token=token)
    result = client.predict(
            message=Tweet_prompt,
            param_2=0.7,
            param_3=True,
            param_4=512,
            api_name="/chat"
    )
    print(result)
    from rich.console import Console
    console = Console(width=80)
    tweet1 = result.split('1:**')[1].split('\n\n')[0]
    tweet2 = result.split('2:**')[1].split('\n\n')[0]
    tweet3 = result.split('3:**')[1]
    console.print(tweet1)
    console.rule()
    console.print(tweet2)
    console.rule()
    console.print(tweet3)
    console.rule()
    return tweet1,tweet2, tweet3

#OR
def createTweets2(token,bruteText):
    # Using https://huggingface.co/spaces/Qwen/Qwen2.5-72B-Instruct
    Tweet_prompt = f"Read the following newsletter. rewrite it into 3 twitter posts in English, in progression.\n---\n{bruteText}"
    client = Client("Qwen/Qwen2.5-72B-Instruct",hf_token=token)
    result = client.predict(
            query=Tweet_prompt,
            history=[],
            system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            api_name="/model_chat"
    )
    twitposts = result[1][0][1]
    console.print(twitposts)

    tweet1 = twitposts.split('Post 1:')[1].split('\n\n')[0]
    tweet2 = twitposts.split('Post 2:')[1].split('\n\n')[0]
    tweet3 = twitposts.split('Post 3:')[1]
    console.print(tweet1)
    console.rule()
    console.print(tweet2)
    console.rule()
    console.print(tweet3)
    console.rule()
    return twitposts


with gr.Blocks(fill_width=True,theme=theme) as demo:
    # INTERFACE
    with gr.Row(variant='panel'):
        with gr.Column(scale=2):
            gr.Image('gradioLOGO.png',width=260)
        with gr.Column(scale=4):
            gr.HTML(
        f"""<h1 style="text-align:center">Advanced POST creation with GRADIO and HF API</h1>""")
            alertTEXT = gr.Text("⚠️✅You don't have a Hugging Face Token set",container=False,show_label=False,)         
        with gr.Column(scale=2):
            TOKEN = gr.Textbox(lines=1,label='Your HF token',scale=1)
            btn_token = gr.Button("Validate HF token", variant='secondary',size='lg',scale=1)
            
             
    with gr.Row(visible=False) as row1:
        #HYPERPARAMETERS
        with gr.Column(scale=1):
            CREATE_SDP = gr.Button(variant='huggingface',value='Generate Prompt')
            GEN_IMAGE = gr.Button(value='Generate Image',variant='primary')
            gr.Markdown('---')
            OPEN_FOLDER = gr.Button(variant='secondary',value='Open Image Folder')
            clear = gr.ClearButton()
        #CHATBOT AREA    
        with gr.Column(scale=3):    
            headers = gr.Textbox(lines=8,label='Header of the Article')
         
    with gr.Row(visible=False) as row2:
        with gr.Column(scale=2):
            SDPrompt = gr.Textbox(lines=8,label='Generated prompt Stable Diffusion')
            ImageFilename = gr.Textbox(lines=2,label='Generated Image Filename',show_copy_button=True)
        with gr.Column(scale=3):
            SDImage = gr.Image(type='pil',label='Generated Image',show_download_button=True, show_fullscreen_button=True,)

    with gr.Row(visible=False) as row3:
        gr.Markdown('---')

    with gr.Row(visible=False) as row4:
        #TWITTERPOSTS CREATION SECTION
        with gr.Column(scale=2):
            body = gr.Textbox(lines=12,label='Body of the Article')
            CREATE_TWEET = gr.Button(variant='huggingface',value='Generate Tweets')
        #TWEET RESULTS AREA    
        with gr.Column(scale=1):    
            tweets1 = gr.Textbox(lines=5,label='🐦 TWEET #1 - 1️⃣',show_copy_button=True)
            tweets2 = gr.Textbox(lines=5,label='🐦 TWEET #2 - 2️⃣',show_copy_button=True)
            tweets3 = gr.Textbox(lines=5,label='🐦 TWEET #3 - 3️⃣',show_copy_button=True)                  

    CREATE_SDP.click(createSDPrompt, [TOKEN,headers], [SDPrompt])
    GEN_IMAGE.click(CreateImage, [TOKEN,SDPrompt], [SDImage,ImageFilename])    #CreateImage
    OPEN_FOLDER.click(openDIR, [], [])    #Open Current directory
    CREATE_TWEET.click(createTweets,[TOKEN,body],[tweets1,tweets2,tweets3])
    btn_token.click(checkHFT,[TOKEN],[row1,row2,row3,row4,alertTEXT])



if __name__ == "__main__":
    demo.launch()   

