# quick_test_predict.py
import os, sys, predict_api

IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else "Capstone Folder"
exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".heic")

imgs = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR) if f.lower().endswith(exts)]
if not imgs:
    raise SystemExit(f"No images found in: {IMG_DIR}")

img = imgs[0]
print("Testing on:", img)

for model in ("VGG16", "ResNet18", "CustomCNN"):
    try:
        res = predict_api.predict(img, model)
        print(model, "→", res)
    except Exception as e:
        print(model, "ERROR:", e)