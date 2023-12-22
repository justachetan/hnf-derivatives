import gdown

url = "https://drive.google.com/drive/folders/1XwzWs5Cz7ymBU4vHyO4p5fmx4Vp7t0rH"
output_path = "../checkpoints"
gdown.download_folder(url, quiet=True, use_cookies=False)