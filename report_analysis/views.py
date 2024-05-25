from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.templatetags.static import static
# from .utils import explanation_pipeline, qa_system
import os
import json
import logging
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.pdfgen.canvas import Canvas

import cv2
import numpy as np
import torch
import pytesseract
from PIL import Image
import re
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(request):
    return render(request, 'report_analysis/main.html')

model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="auto",trust_remote_code=False,revision="main")
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
    logging.getLogger("transformers").setLevel(logging.ERROR)
    for i, sent in enumerate(sequences):
        if i == 0:
            prompt = f"<s>[INST] Explain the following medical text and its jargons in simple English very concisely: {sent} [/INST]"
        else:
            prompt = f"[INST] Next, explain in simple terms for a non-medical audience: {sent} [/INST]"

        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        output = model.generate(
            input_ids, temperature=0.6, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=10)
        input_length = input_ids.size(1)
        generated_tokens = output[:, input_length:]
        output_str += tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True)
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

def custom_canvas(canvas, doc):
    page_width, page_height = letter
    border_margin = 3 * mm
    canvas.setStrokeColorRGB(0, 0, 0)
    canvas.rect(border_margin, border_margin, page_width - 2 *
                border_margin, page_height - 2 * border_margin)
    page_num_text = f"{canvas.getPageNumber()}"
    canvas.drawCentredString(page_width / 2.0, 12 * mm, page_num_text)
    # logo_path = "D:/Downloads/BE_project(MAIN)/BE_project_nlp(MAIN)/BE project/drsimplify/logo.png"
    logo_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "report_analysis/static/report_analysis/img/logo.png")
    logo_width = 1.45 * inch
    logo_height = 0.32 * inch
    canvas.drawImage(logo_path, page_width - logo_width - 72 + (0.65 * inch), page_height -
                     logo_height - 95 + inch, width=logo_width, height=logo_height, mask='auto')


def export_to_pdf(content, filename="Medical_explanations.pdf"):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72,
                            leftMargin=72, topMargin=72, bottomMargin=18)
    styles = getSampleStyleSheet()

    title_style = styles['Title']
    title_style.alignment = 1
    title_style.spaceAfter = 20

    header_style = ParagraphStyle(
        'header_style', parent=styles['Heading1'], fontSize=14, leading=20, spaceAfter=10, alignment=1, textColor=colors.purple)
    content_style = styles['BodyText']
    content_style.fontSize = 12
    content_style.leading = 15

    story = [Paragraph("In-depth Medical Report Analysis", title_style),
             Paragraph("Comprehensive Explanation", header_style),
             Spacer(1, 12)]

    for paragraph in content.split('\n\n'):
        story.append(Paragraph(paragraph, content_style))
        story.append(Spacer(1, 12))

    doc.build(story, onFirstPage=custom_canvas, onLaterPages=custom_canvas)
    buffer.seek(0)
    return buffer


@csrf_exempt
def upload_and_explain(request):
    if request.method == 'POST' and request.FILES:
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'error': 'No file provided.'}, status=400)

        try:
            file_path = file.temporary_file_path()
        except AttributeError:
            with ContentFile(file.read()) as temp_file:
                file_path = default_storage.save("tmp/somefile.tmp", temp_file)

        explanation = explanation_pipeline(file_path)
        buffer = export_to_pdf(explanation)
        if file_path and not file_path.startswith("/tmp"):
            default_storage.delete(file_path)

        try:
            return FileResponse(buffer, as_attachment=True, filename='Medical_Report_Explanation.pdf')
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def ask_question(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        if question:
            answer = qa_system(question)
            return JsonResponse({'answer': answer})
        else:
            return JsonResponse({'error': 'Empty question received'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method. Use POST.'}, status=405)


# OLD WORKING WITHOUT CONTEXT MANAGER
# @csrf_exempt
# def upload_and_explain(request):
#     if request.method == 'POST' and request.FILES:
#         file = request.FILES.get('file')
#         if not file:
#             return JsonResponse({'error': 'No file provided.'}, status=400)

#         try:
#             file_path = file.temporary_file_path()
#         except AttributeError:
#             temp_file = ContentFile(file.read())
#             file_path = default_storage.save("tmp/somefile.tmp", temp_file)

#         explanation = explanation_pipeline(file_path, "")
#         buffer = export_to_pdf(explanation)
#         try:
#             return FileResponse(
#                 buffer, as_attachment=True, filename='Medical_Report_Explanation.pdf')
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)


# OLD NOT WORKING
# @csrf_exempt
# def upload_and_explain(request):
#     if request.method == 'POST' and request.FILES:
#         file = request.FILES.get('file')
#         if not file:
#             return JsonResponse({'error': 'No file provided.'}, status=400)

#         try:
#             file_path = file.temporary_file_path()
#         except AttributeError:
#             temp_file = ContentFile(file.read())
#             file_path = default_storage.save("tmp/somefile.tmp", temp_file)

#         explanation = explanation_pipeline(file_path, "")
#         pdf_path = export_to_pdf(
#             explanation, "dcrsimplify/tmp/Medical_explanations.pdf")
#         try:
#             with open(pdf_path, 'rb') as f:
#                 response = FileResponse(
#                     f, as_attachment=True, filename='Medical_Report_Explanation.pdf')
#             os.remove(pdf_path)
#             return response
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=500)
