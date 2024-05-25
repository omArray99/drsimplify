from django.conf import settings
import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image
import re
from typing import List
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfgen import canvas

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",trust_remote_code=False,revision="main")

# model_name = "openai-community/gpt2"
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, device_map="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


def preprocess_image(image):
    np_image = np.array(image)
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    thresholded_img = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 275, 60)
    return thresholded_img


def extract_text(image):
    return pytesseract.image_to_string(image)


def preprocess_text(text):
    text = text.lower()
    allowed_chars = "abcdefghijklmnopqrstuvwxyz0123456789 ,.-()/\%+*=\n;"
    text = ''.join(c for c in text if c in allowed_chars)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*\n\s*', '\n', text)
    return text


def split_into_sentences(paragraph: str) -> List[str]:
    sentence_endings = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z])", re.IGNORECASE)
    sentences = sentence_endings.split(paragraph)
    return sentences


def group_sentences(sentences: List[str], group_size: int = 3) -> List[str]:
    grouped_sentences = [' '.join(sentences[i:i + group_size])
                         for i in range(0, len(sentences), group_size)]
    return grouped_sentences


def process_image(image_path):
    image = Image.open(image_path)
    processed_image = preprocess_image(image)
    extracted_text = extract_text(processed_image)
    preprocessed_text = preprocess_text(extracted_text)
    sentences = split_into_sentences(preprocessed_text)
    grouped = group_sentences(sentences)
    return grouped


def explanation_pipeline(image_path, output_str=""):
    sequences = process_image(image_path)
   #  explanations = []
    logging.getLogger("transformers").setLevel(logging.ERROR)
    for i, sent in enumerate(sequences):
        if i == 0:
            prompt = f"<s>[INST] Explain the following medical text and its jargons in simple English very concisely: {sent} [/INST]"
        else:
            prompt = f"[INST] Next, explain in simple terms for a non-medical audience: {sent} [/INST]"

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = model.generate(
            input_ids, temperature=0.6, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=250)
        input_length = input_ids.size(1)
        generated_tokens = output[:, input_length:]
        output_str += tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True)
      #   explanations.append(explanation)
      #   final_content = " ".join(explanations)
      #   print(f"Final content to be converted to PDF: {final_content}")
      #   logging.debug(f"Final content to be converted to PDF: {final_content}")
    return output_str


def qa_system(query):
    prompt = f'''<s>[INST]You are a medical chat language model. You will only answer medical-related questions and refuse any and all non-medical questions [/INST]\n\n[INST] Question: {query} [/INST]'''
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7,
                            do_sample=True, top_p=0.95, top_k=40, max_new_tokens=250)
    input_length = input_ids.size(1)
    generated_tokens = output[:, input_length:]
    generated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated


# def custom_canvas(canvas, doc):
#     page_width, page_height = letter
#     border_margin = 3 * mm
#     canvas.setStrokeColorRGB(0, 0, 0)
#     canvas.rect(border_margin, border_margin, page_width - 2 *
#                 border_margin, page_height - 2 * border_margin)
#     page_num_text = f"{canvas.getPageNumber()}"
#     canvas.drawCentredString(page_width / 2.0, 12 * mm, page_num_text)
#     logo_path = 'D:/Downloads/BE_project(MAIN)/BE_project_nlp(MAIN)/BE project/drsimplify/logo.png'
#     logo_width = 1.45 * inch
#     logo_height = 0.32 * inch
#     try:
#         canvas.drawImage(logo_path, page_width - logo_width - 72 + (0.65 * inch), page_height -
#                          logo_height - 95 + inch, width=logo_width, height=logo_height, mask='auto')
#     except Exception as e:
#         logging.error(f"Failed to load or draw logo: {str(e)}")


# def export_to_pdf(content, filename="D:/Poems/Medical_Report_Explanation.pdf"):
#     try:
#         if not os.path.exists(os.path.dirname(filename)):
#             os.makedirs(os.path.dirname(filename))

#         doc = SimpleDocTemplate(filename, pagesize=letter)
#         styles = getSampleStyleSheet()

#         title_style = styles['Title']
#         title_style.alignment = 1
#         title_style.spaceAfter = 20

#         header_style = ParagraphStyle(
#             'header_style', parent=styles['Heading1'], fontSize=14, leading=20, spaceAfter=10, alignment=1, textColor=colors.purple)
#         content_style = styles['BodyText']
#         content_style.fontSize = 12
#         content_style.leading = 15

#         story = [Paragraph("In-depth Medical Report Analysis", title_style),
#                  Paragraph("Comprehensive Explanation", header_style),
#                  Spacer(1, 12)]

#         for paragraph in content.split('\n\n'):
#             story.append(Paragraph(paragraph, content_style))
#             story.append(Spacer(1, 12))

#         doc.build(story, onFirstPage=custom_canvas, onLaterPages=custom_canvas)
#       #   return filename
#     except Exception as e:
#         logging.error(f"Failed to generate PDF: {str(e)}")
#         raise Exception(
#             f"An error occurred while generating the PDF: {str(e)}")
