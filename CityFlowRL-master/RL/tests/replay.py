import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


def run(config):
    homedir = os.path.expanduser("~")
    webdriver_service = Service(f"{homedir}/chromedriver/stable/chromedriver")
    options = webdriver.ChromeOptions()
    # options.add_argument = {'user-data-dir':x '/Users/Application/Chrome/Default'}
    options.add_argument("--kiosk")
    options.add_argument("--disable-infobars")
    driver = webdriver.Chrome(service=webdriver_service, options=options)
    file = os.path.dirname(os.getcwd()) + "/frontend/index.html"
    driver.get('file://' + file)
    time.sleep(1)
    driver.find_element(By.ID, 'roadnet-file').send_keys(os.path.join(os.getcwd(), f'configs/{config}/roadnetLog.json'))
    driver.find_element(By.ID, 'replay-file').send_keys(os.path.join(os.getcwd(), f'configs/{config}/replayLog.txt'))
    driver.find_element(By.ID, 'start-btn').click()
    time.sleep(10)
    el = driver.find_element(By.ID, 'current-step-num').get_attribute('innerText')
    time.sleep(1)
    newEl = driver.find_element(By.ID, 'current-step-num').get_attribute('innerText')
    while el != newEl:
        el = driver.find_element(By.ID, 'current-step-num').get_attribute('innerText')
        time.sleep(1)
        newEl = driver.find_element(By.ID, 'current-step-num').get_attribute('innerText')
    driver.quit()


if __name__ == '__main__':
    run('hangzhou_1x1_bc-tyc_18041607_1h')
