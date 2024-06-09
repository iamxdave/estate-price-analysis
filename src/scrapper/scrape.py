from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin, urlparse
import time

def scrape(url=None, body_only=True, wait_seconds=0):
    url = urljoin(url, urlparse(url).path)
    print(f'Navigating to {url}')

    # Set up options for the driver
    options = Options()
    options.add_argument("--headless")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')

    # Set up the driver
    webdriver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=webdriver_service, options=options)

    # Go to the URL
    driver.get(url)

    # Wait for the necessary elements to load
    WebDriverWait(driver, wait_seconds).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

    print('Navigated! Scraping page content...')

    # Get the HTML of the body
    html = driver.page_source
    if body_only:
        body = driver.find_element(By.TAG_NAME, "body")
        html = body.get_attribute('innerHTML')

    # Close the driver
    driver.quit()

    return html

def scrape_pagination(url=None, body_only=True, wait_seconds=0, times=1):
    url = urljoin(url, urlparse(url).path)

    # Set up options for the driver
    options = Options()
    options.add_argument("--headless")
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')

    # Set up the driver
    webdriver_service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=webdriver_service, options=options)
    
    paged_html = []
    for i in range(times):
        page_url = f"{url}?page={i + 1}"
        
        print(f'Navigating to {page_url}')

        # Wait for the necessary elements to load
        WebDriverWait(driver, wait_seconds).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        # Go to the next URL
        driver.get(page_url)

        print('Navigated! Scraping page content...')

        # Get the HTML of the body
        html = driver.page_source
        if body_only:
            body = driver.find_element(By.TAG_NAME, "body")
            html = body.get_attribute('innerHTML')

        paged_html.append(html)

        time.sleep(wait_seconds)

    print('Scarped all pages!')

    # Close the driver
    driver.quit()

    return paged_html