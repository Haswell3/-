from django.contrib import admin
from django.urls import path

from django.conf import settings
from django.conf.urls.static import static
from whisper import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('media/', views.upload_audio, name='upload_audio'),
    path('rag/', views.upload_text, name='upload_text')
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
