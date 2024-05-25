from django.urls import path
from .views import main, upload_and_explain, ask_question

urlpatterns = [
    path('', main, name="home_page"),
    path('upload-explain/', upload_and_explain, name='upload_and_explain'),
    path('ask-question/', ask_question, name='ask_question'),
]
