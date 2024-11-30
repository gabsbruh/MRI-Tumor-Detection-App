from selenium import webdriver
from selenium.webdriver.common.by import By

def test_e2e_single_detection():
    driver = webdriver.Chrome()
    driver.get("http://127.0.0.1:5000")  # Replace with your app's URL
    driver.find_element(By.XPATH, "//button[text()='Single MRI']").click()
    driver.find_element(By.ID, "file-upload").send_keys("/path/to/image.jpg")
    driver.find_element(By.ID, "detect-btn").click()
    assert "DETECTED" in driver.page_source or "No Tumor" in driver.page_source
    driver.quit()