from django.db import models

class UploadedAudio(models.Model):
    audio = models.FileField(upload_to='uploads/')


class UploadedTextFile(models.Model):
    text_file = models.FileField(upload_to='output/')


