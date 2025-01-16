import asyncio
import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from browser_use.browser.context import BrowserContextWindowSize
from datetime import datetime
from pydantic import BaseModel
from typing import List

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Create images directory in current folder
images_dir = Path("generated_images")
images_dir.mkdir(exist_ok=True)

# Initialize browser with real Chrome and proper context config
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
controller = Controller()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key)
)

class Model(BaseModel):
	title: str
	url: str
	likes: int
	license: str


class Models(BaseModel):
	models: List[Model]


@controller.action('Save models', param_model=Models)
def save_models(params: Models):
	with open('models.txt', 'a') as f:
		for model in params.models:
			f.write(f'{model.title} ({model.url}): {model.likes} likes, {model.license}\n')


async def main():

    agent = Agent(
        task=(
            'Look up models with a license of cc-by-sa-4.0 and sort by most likes on Hugging face, save top 5 to file.'
        ),
        llm=llm,
        controller=controller,
        browser=browser,
    )

    # Start the agent
    await agent.run()

    await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
