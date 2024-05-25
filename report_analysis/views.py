from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings
from django.templatetags.static import static
from .utils import explanation_pipeline, qa_system
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


def main(request):
    return render(request, 'report_analysis/main.html')


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
