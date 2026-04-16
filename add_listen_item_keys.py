"""Add Listen Lab per-item tooltip/status keys to translations.json."""
import json, os

PATH = os.path.join(os.path.dirname(__file__), "translations.json")
with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

NEW = {
  "LISTEN_LAB_TOOLTIP_REMOVE_ITEM": {
    "value":"Remove from list","en":"Remove from list",
    "es":"Eliminar de la lista","fr":"Supprimer de la liste","de":"Aus der Liste entfernen",
    "pt-BR":"Remover da lista","it":"Rimuovi dall'elenco","nl":"Verwijderen uit lijst",
    "ru":"Удалить из списка","ja":"リストから削除","zh-CN":"从列表中删除",
    "ko":"목록에서 제거","pl":"Usuń z listy","tr":"Listeden kaldır",
    "ar":"إزالة من القائمة","hi":"सूची से हटाएं","sv":"Ta bort från listan"
  },
  "LISTEN_LAB_TOOLTIP_CANCEL_CONV": {
    "value":"Cancel conversion","en":"Cancel conversion",
    "es":"Cancelar conversión","fr":"Annuler la conversion","de":"Konvertierung abbrechen",
    "pt-BR":"Cancelar conversão","it":"Annulla la conversione","nl":"Conversie annuleren",
    "ru":"Отменить конвертацию","ja":"変換をキャンセル","zh-CN":"取消转换",
    "ko":"변환 취소","pl":"Anuluj konwersję","tr":"Dönüşümü iptal et",
    "ar":"إلغاء التحويل","hi":"कन्वर्ज़न रद्द करें","sv":"Avbryt konvertering"
  },
  "LISTEN_LAB_TOOLTIP_RESUME": {
    "value":"Resume","en":"Resume","es":"Reanudar","fr":"Reprendre","de":"Fortsetzen",
    "pt-BR":"Retomar","it":"Riprendi","nl":"Hervatten","ru":"Возобновить",
    "ja":"再開","zh-CN":"继续","ko":"재개","pl":"Wznów","tr":"Devam et",
    "ar":"استئناف","hi":"फिर शुरू करें","sv":"Återuppta"
  },
  "LISTEN_LAB_TOOLTIP_PAUSE_CONV": {
    "value":"Pause conversion","en":"Pause conversion",
    "es":"Pausar conversión","fr":"Mettre la conversion en pause",
    "de":"Konvertierung pausieren","pt-BR":"Pausar conversão",
    "it":"Metti in pausa la conversione","nl":"Conversie pauzeren",
    "ru":"Приостановить конвертацию","ja":"変換を一時停止","zh-CN":"暂停转换",
    "ko":"변환 일시 정지","pl":"Wstrzymaj konwersję","tr":"Dönüşümü duraklat",
    "ar":"إيقاف التحويل مؤقتًا","hi":"कन्वर्ज़न रोकें","sv":"Pausa konvertering"
  },
  "LISTEN_LAB_TOOLTIP_EDIT_META": {
    "value":"Edit audio metadata","en":"Edit audio metadata",
    "es":"Editar metadatos de audio","fr":"Modifier les métadonnées audio",
    "de":"Audio-Metadaten bearbeiten","pt-BR":"Editar metadados de áudio",
    "it":"Modifica metadati audio","nl":"Audiometadata bewerken",
    "ru":"Редактировать метаданные аудио","ja":"オーディオメタデータを編集",
    "zh-CN":"编辑音频元数据","ko":"오디오 메타데이터 편집",
    "pl":"Edytuj metadane audio","tr":"Ses meta verilerini düzenle",
    "ar":"تعديل بيانات الصوت","hi":"ऑडियो मेटाडेटा संपादित करें","sv":"Redigera ljudmetadata"
  },
  "LISTEN_LAB_TOOLTIP_SAVE_AUDIO": {
    "value":"Save translated audio","en":"Save translated audio",
    "es":"Guardar audio traducido","fr":"Enregistrer l'audio traduit",
    "de":"Übersetztes Audio speichern","pt-BR":"Salvar áudio traduzido",
    "it":"Salva audio tradotto","nl":"Vertaald audio opslaan",
    "ru":"Сохранить переведённое аудио","ja":"翻訳済み音声を保存",
    "zh-CN":"保存已翻译的音频","ko":"번역된 오디오 저장",
    "pl":"Zapisz przetłumaczone audio","tr":"Çevrilmiş sesi kaydet",
    "ar":"حفظ الصوت المترجم","hi":"अनुवादित ऑडियो सेव करें","sv":"Spara översatt ljud"
  },
  "LISTEN_LAB_TOOLTIP_NEXT_CH": {
    "value":"Next chapter","en":"Next chapter",
    "es":"Siguiente capítulo","fr":"Chapitre suivant","de":"Nächstes Kapitel",
    "pt-BR":"Próximo capítulo","it":"Capitolo successivo","nl":"Volgend hoofdstuk",
    "ru":"Следующая глава","ja":"次のチャプター","zh-CN":"下一章","ko":"다음 챕터",
    "pl":"Następny rozdział","tr":"Sonraki bölüm","ar":"الفصل التالي",
    "hi":"अगला अध्याय","sv":"Nästa kapitel"
  },
  "LISTEN_LAB_TOOLTIP_PREV_CH": {
    "value":"Previous chapter","en":"Previous chapter",
    "es":"Capítulo anterior","fr":"Chapitre précédent","de":"Vorheriges Kapitel",
    "pt-BR":"Capítulo anterior","it":"Capitolo precedente","nl":"Vorig hoofdstuk",
    "ru":"Предыдущая глава","ja":"前のチャプター","zh-CN":"上一章","ko":"이전 챕터",
    "pl":"Poprzedni rozdział","tr":"Önceki bölüm","ar":"الفصل السابق",
    "hi":"पिछला अध्याय","sv":"Föregående kapitel"
  },
  "LISTEN_LAB_TOOLTIP_SKIP_FWD": {
    "value":"Skip forward 30 seconds","en":"Skip forward 30 seconds",
    "es":"Saltar 30 segundos adelante","fr":"Avancer de 30 secondes",
    "de":"30 Sekunden vorspulen","pt-BR":"Avançar 30 segundos",
    "it":"Salta avanti di 30 secondi","nl":"30 seconden vooruit spoelen",
    "ru":"Перемотать вперёд на 30 секунд","ja":"30秒スキップ","zh-CN":"快进 30 秒",
    "ko":"30초 앞으로 건너뛰기","pl":"Przewiń do przodu 30 sekund",
    "tr":"30 saniye ileri atla","ar":"تخطي 30 ثانية للأمام",
    "hi":"30 सेकंड आगे बढ़ें","sv":"Hoppa framåt 30 sekunder"
  },
  "LISTEN_LAB_TOOLTIP_REWIND": {
    "value":"Rewind 30 seconds","en":"Rewind 30 seconds",
    "es":"Retroceder 30 segundos","fr":"Reculer de 30 secondes",
    "de":"30 Sekunden zurückspulen","pt-BR":"Retroceder 30 segundos",
    "it":"Riavvolgi di 30 secondi","nl":"30 seconden terugspoelen",
    "ru":"Перемотать назад на 30 секунд","ja":"30秒巻き戻し","zh-CN":"后退 30 秒",
    "ko":"30초 뒤로 되감기","pl":"Przewiń do tyłu 30 sekund",
    "tr":"30 saniye geri al","ar":"إرجاع 30 ثانية",
    "hi":"30 सेकंड पीछे जाएं","sv":"Spola tillbaka 30 sekunder"
  },
  "LISTEN_LAB_TOOLTIP_PAUSE": {
    "value":"Pause","en":"Pause","es":"Pausar","fr":"Pause","de":"Pausieren",
    "pt-BR":"Pausar","it":"Pausa","nl":"Pauzeren","ru":"Пауза","ja":"一時停止",
    "zh-CN":"暂停","ko":"일시 정지","pl":"Wstrzymaj","tr":"Duraklat",
    "ar":"إيقاف مؤقت","hi":"रोकें","sv":"Pausa"
  },
  "LISTEN_LAB_TOOLTIP_PLAY_AUDIO": {
    "value":"Play audio","en":"Play audio",
    "es":"Reproducir audio","fr":"Lire l'audio","de":"Audio abspielen",
    "pt-BR":"Reproduzir áudio","it":"Riproduci audio","nl":"Audio afspelen",
    "ru":"Воспроизвести аудио","ja":"オーディオを再生","zh-CN":"播放音频",
    "ko":"오디오 재생","pl":"Odtwórz audio","tr":"Ses oynat",
    "ar":"تشغيل الصوت","hi":"ऑडियो चलाएं","sv":"Spela upp ljud"
  },
  "LISTEN_LAB_TOOLTIP_PLAY_TRANSLATED": {
    "value":"Play translated audio","en":"Play translated audio",
    "es":"Reproducir audio traducido","fr":"Lire l'audio traduit",
    "de":"Übersetztes Audio abspielen","pt-BR":"Reproduzir áudio traduzido",
    "it":"Riproduci audio tradotto","nl":"Vertaald audio afspelen",
    "ru":"Воспроизвести переведённое аудио","ja":"翻訳済み音声を再生",
    "zh-CN":"播放已翻译的音频","ko":"번역된 오디오 재생",
    "pl":"Odtwórz przetłumaczone audio","tr":"Çevrilmiş sesi oynat",
    "ar":"تشغيل الصوت المترجم","hi":"अनुवादित ऑडियो चलाएं","sv":"Spela upp översatt ljud"
  },
  # Listen Lab active item status labels
  "LISTEN_LAB_STATUS_TRANSCRIBING": {
    "value":"🎙  Transcribing…","en":"🎙  Transcribing…",
    "es":"🎙  Transcribiendo…","fr":"🎙  Transcription…","de":"🎙  Transkribieren…",
    "pt-BR":"🎙  Transcrevendo…","it":"🎙  Trascrizione…","nl":"🎙  Transcriberen…",
    "ru":"🎙  Транскрибирование…","ja":"🎙  文字起こし中…","zh-CN":"🎙  转录中…",
    "ko":"🎙  전사 중…","pl":"🎙  Transkrypcja…","tr":"🎙  Transkripsiyon…",
    "ar":"🎙  جارٍ النسخ…","hi":"🎙  ट्रांसक्राइब हो रहा है…","sv":"🎙  Transkriberar…"
  },
  "LISTEN_LAB_STATUS_TRANSLATING": {
    "value":"🌐  Translating…","en":"🌐  Translating…",
    "es":"🌐  Traduciendo…","fr":"🌐  Traduction…","de":"🌐  Übersetzen…",
    "pt-BR":"🌐  Traduzindo…","it":"🌐  Traduzione…","nl":"🌐  Vertalen…",
    "ru":"🌐  Перевод…","ja":"🌐  翻訳中…","zh-CN":"🌐  翻译中…",
    "ko":"🌐  번역 중…","pl":"🌐  Tłumaczenie…","tr":"🌐  Çeviriliyor…",
    "ar":"🌐  جارٍ الترجمة…","hi":"🌐  अनुवाद हो रहा है…","sv":"🌐  Översätter…"
  },
  "LISTEN_LAB_STATUS_CONVERTING": {
    "value":"🔊  Converting…","en":"🔊  Converting…",
    "es":"🔊  Convirtiendo…","fr":"🔊  Conversion…","de":"🔊  Konvertieren…",
    "pt-BR":"🔊  Convertendo…","it":"🔊  Conversione…","nl":"🔊  Converteren…",
    "ru":"🔊  Конвертация…","ja":"🔊  変換中…","zh-CN":"🔊  转换中…",
    "ko":"🔊  변환 중…","pl":"🔊  Konwersja…","tr":"🔊  Dönüştürülüyor…",
    "ar":"🔊  جارٍ التحويل…","hi":"🔊  कन्वर्ट हो रहा है…","sv":"🔊  Konverterar…"
  },
}

added = 0
for key, val in NEW.items():
    if key not in data["strings"]:
        data["strings"][key] = val
        added += 1

with open(PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done. Added {added} keys. Total: {len(data['strings'])}")
