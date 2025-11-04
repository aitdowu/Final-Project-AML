import fitz  # PyMuPDF
import os
from tqdm import tqdm

# List your PDF filenames here
pdf_files = [
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week1-2.pdf",
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week2-2.pdf",
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week3-2.pdf",
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week4-2.pdf",
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week5.pdf",
    "/Users/tanki/Final Project AML/rag_course_notes_chatbot/data/Week7.pdf"  # this will actually be your Week6 file
]

# Output folder for extracted images
os.makedirs("extracted_images", exist_ok=True)

for pdf_path in pdf_files:
    if not os.path.exists(pdf_path):
        print(f"⚠️ Skipping {pdf_path} (not found)")
        continue

    doc = fitz.open(pdf_path)
    week_name = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = os.path.join("extracted_images", week_name)
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

print("\n✅ Done! All images saved in the 'extracted_images' folder.")