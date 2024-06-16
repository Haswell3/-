from django import forms
from .models import UploadedAudio
from .models import UploadedTextFile
from django.core.validators import FileExtensionValidator

class AudioUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedAudio
        fields = ('audio',)

    audio = forms.FileField(label='音声ファイル')


class TextFileUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedTextFile  # .txt ファイルを保存するモデルに変更する
        fields = ('text_file',)  # .txt ファイルを保存するフィールドに変更する

    text_file = forms.FileField(
        label='テキストファイル（.txt）', 
        validators=[FileExtensionValidator(allowed_extensions=['txt'])]  # 拡張子が .txt のファイルのみを受け入れるように設定する
    )