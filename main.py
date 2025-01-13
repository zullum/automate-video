import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig
from browser_use.browser.context import BrowserContextWindowSize

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')

# Create downloads directory if it doesn't exist
downloads_dir = Path.home() / "Downloads" / "browser_use_downloads"
downloads_dir.mkdir(parents=True, exist_ok=True)

# Track downloads
downloaded_files = []

async def handle_download(download):
    print("Download started!")
    # Wait for the download to complete
    path = await download.path()
    print(f"Download completed: {path}")
    if path:
        # Create new path in our downloads directory
        new_path = downloads_dir / os.path.basename(path)
        # Move the file
        os.rename(path, new_path)
        downloaded_files.append(str(new_path))
        print(f"Downloaded and moved to: {new_path}")

# Initialize browser with real Chrome and proper context config
browser = Browser(
    config=BrowserConfig(
        headless=False,
        disable_security=True,
        new_context_config=BrowserContextConfig(
            disable_security=True,
            browser_window_size=BrowserContextWindowSize(width=1280, height=1100),
            minimum_wait_page_load_time=4,
            maximum_wait_page_load_time=20,
        ),
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    )
)
controller = Controller()

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-exp',
    api_key=SecretStr(api_key)
)

async def run_image_generation():
    # Get the playwright browser instance
    playwright_browser = await browser.get_playwright_browser()
    print("Got playwright browser")
    
    # Create a new context and page
    context = await playwright_browser.new_context(accept_downloads=True)
    page = await context.new_page()
    
    # Set up download handler
    page.on("download", lambda download: asyncio.create_task(handle_download(download)))
    print("Download handler set up")
    
    agent = Agent(
        task=(
            'Go to https://labs.google/fx/tools/image-fx and wait for interface to load. '
            'Enter the prompt "Create an image of an angry lion in savanna in animated style" '
            'Click on "Create" '
            'Wait for the image to be generated. '
            'Download the first image'
        ),
        llm=llm,
        controller=controller,
        browser=browser,
    )

    await agent.run()
    
    # Close everything
    await context.close()
    await browser.close()

    # Show downloaded files
    if downloaded_files:
        print("\nDownloaded files:")
        for file_path in downloaded_files:
            print(f"- {os.path.basename(file_path)}")
            print(f"  Full path: {file_path}")
    else:
        print("\nNo files were downloaded during this session.")

    input('Press Enter to close...')

if __name__ == '__main__':
    asyncio.run(run_image_generation())
