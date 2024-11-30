from selenium import webdriver
from selenium.webdriver.common.by import By
import os

def test_e2e_batch_image_detection():
    """Simulate the batch detection workflow."""
    driver = webdriver.Chrome()
    driver.get("http://127.0.0.1:5000")  # Replace with your app's URL

    # Navigate to the Multiple MRI page
    driver.find_element(By.XPATH, "//button[text()='Multiple MRI']").click()

    # Mock file upload (adjust file selector ID/class based on your UI)
    file_input = driver.find_element(By.ID, "folder-input")
    file_input.send_keys("/path/to/test_images_folder")

    # Click "Detect" button
    detect_button = driver.find_element(By.ID, "detect-btn")
    detect_button.click()

    # Wait for the results to populate
    driver.implicitly_wait(10)

    # Verify detection results are displayed in the table
    results_table = driver.find_element(By.ID, "results-table")
    assert results_table.is_displayed()

    # Export results to a CSV
    export_button = driver.find_element(By.ID, "export-csv-btn")
    export_button.click()

    # Verify the exported CSV file exists
    exported_csv = "/path/to/exported_results.csv"
    assert os.path.exists(exported_csv)

    driver.quit()