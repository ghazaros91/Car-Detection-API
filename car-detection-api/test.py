import requests

def test_analyze_image(image_path: str):
    url = "http://localhost:8000/analyze-image/"

    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        response = requests.post(url, files=files)

    print("Status:", response.status_code)
    try:
        print("Response:", response.json())
    except Exception as e:
        print("Failed to parse JSON:", e)
        print("Raw response:", response.text)

if __name__ == "__main__":
    test_analyze_image("images/image.png")