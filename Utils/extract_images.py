import fitz  # PyMuPDF
import os
from tqdm import tqdm

# Get project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, "data")
extracted_images_dir = os.path.join(project_root, "extracted_images")

# List your PDF filenames here (relative to data directory)
pdf_files = [
    "Week7(1)-2.pdf",
    "Week8-2.pdf", 
    "Week9.pdf",
]

# Output folder for extracted images
os.makedirs(extracted_images_dir, exist_ok=True)

for pdf_filename in pdf_files:
    pdf_path = os.path.join(data_dir, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"⚠️ Skipping {pdf_path} (not found)")
        continue

    doc = fitz.open(pdf_path)
    week_name = os.path.splitext(pdf_filename)[0]
    out_dir = os.path.join(extracted_images_dir, week_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"📘 Extracting images from {pdf_path} ...")
    for page_index in tqdm(range(len(doc)), desc=week_name):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{week_name}_page{page_index+1}_img{img_index}.{image_ext}"
            with open(os.path.join(out_dir, image_filename), "wb") as f:
                f.write(image_bytes)

print(f"\n✅ Done! All images saved in the '{extracted_images_dir}' folder.")

