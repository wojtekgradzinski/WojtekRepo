from selenium import webdriver
from selenium.webdriver.common.by import By

def get_cpf():
    driver = webdriver.Chrome("C:/Users/rajat/Downloads/chromedriver")
    driver.get("https://www.yelp.co.uk/biz/stadtklause-berlin")
    css_selector = "button[class = ' css-174jypu']"
    driver.find_element_by_css_selector(css_selector).click()
    time.sleep(np.random.randint(5,10))
    outer_div = "arrange__373c0__UHqhV gutter-2__373c0__3Zpeq layout-wrap__373c0__34d4b layout-2-units__373c0__3CiAk border-color--default__373c0__2oFDT"
    text = driver.find_elements_by_class_name(outer_div)
    print(text)

get_cpf()