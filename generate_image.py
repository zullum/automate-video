import asyncio
import os
import base64
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from browser_use.browser.context import BrowserContextWindowSize
from datetime import datetime
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Loading environment variables")
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError('GEMINI_API_KEY is not set')

# Create images directory in current folder
images_dir = Path("generated_images")
images_dir.mkdir(exist_ok=True)
logger.info(f"Created/verified images directory at {images_dir}")

# Initialize browser with real Chrome and proper context config
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig(
            disable_security=True,
        )
    )
)
controller = Controller()
logger.info("Browser and controller initialized")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key)
)
logger.info("LLM initialized")

# Track downloads
downloaded_files: List[str] = []

async def handle_download(download):
    logger.info("Download event triggered")
    # Get the executable directory
    exe_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Get original download path
    original_path = await download.path()
    if original_path:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_image_{timestamp}.png"
        
        # Create new path in images directory
        new_path = images_dir / filename

        logger.debug(f"Original path: {original_path}")
        logger.debug(f"New path: {new_path}")
        
        # Move the file to images directory
        os.rename(original_path, str(new_path))
        
        # Add the new path to downloaded files list
        downloaded_files.append(str(new_path))
        logger.info(f"Image downloaded and moved to: {new_path}")

def handle_page(new_page):
    logger.info("New page created")
    new_page.on("download", lambda download: asyncio.create_task(handle_download(download)))

async def setup_browser_handlers(playwright_browser):
    logger.info("Setting up browser handlers...")
    try:
        for context in playwright_browser.contexts:
            context.on("page", handle_page)
            for page in context.pages:
                page.on("download", lambda download: asyncio.create_task(handle_download(download)))
        logger.info("Browser handlers set up successfully")
    except Exception as e:
        logger.error(f"Error setting up browser handlers: {e}")
        raise

async def run_image_generation(image_prompt: str = "Create an image of an angry lion in savanna in animated style"):
    logger.info(f"Starting image generation with prompt: {image_prompt}")
    agent = Agent(
        task=(
            'Go to https://labs.google/fx/tools/image-fx or open it if it does not exist and wait for interface to load. '
            'login with your google account user: zullumsan@gmail.com password: tubeedavoxis1987'
            'Find conteneditable div with image prompt click on it'
            f'Enter the prompt "{image_prompt}" '
            'Click on "Create" '
            'Wait for the image to be generated. '
            'When the image is generated, find the first image download it.'
        ),
        llm=llm,
        controller=controller,
        browser=browser,
    )

    try:
        # Get the underlying Playwright browser instance
        playwright_browser = await browser.get_playwright_browser()
        logger.info(f'Initial browser contexts: {len(playwright_browser.contexts)}')
        
        # Set up handlers
        await setup_browser_handlers(playwright_browser)
        # Run the agent
        logger.info("Starting agent execution")
        await agent.run()
        logger.info("Agent execution completed")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        logger.info("Closing browser")
        await browser.close()

    # Show saved files
    saved_files = list(images_dir.glob('*.png'))
    if saved_files:
        logger.info("\nSaved images:")
        for file_path in saved_files:
            logger.info(f"- {file_path}")
    else:
        logger.warning("No images were saved during this session.")

    return downloaded_files

if __name__ == '__main__':
    # Example usage with custom prompt
    custom_prompt = "Create an image of a peaceful sunset over mountains in watercolor style"
    asyncio.run(run_image_generation(custom_prompt))
