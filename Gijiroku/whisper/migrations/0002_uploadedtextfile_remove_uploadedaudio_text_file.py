# Generated by Django 4.2.5 on 2023-11-03 14:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('whisper', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedTextFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text_file', models.FileField(upload_to='output/')),
            ],
        )
    ]
