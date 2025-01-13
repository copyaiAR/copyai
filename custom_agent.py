import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(file_path, mime_type=None):
  file = genai.upload_file(file_path, mime_type=mime_type)
  return file

def get_gemini_response(file_path):
  """
  يقوم بتحميل ملف إلى Gemini والحصول على وصف تفصيلي له.

  Args:
    file_path: مسار الملف المراد وصفه.

  Returns:
    google.generativeai.types.generation_types.GenerateContentResponse: كائن الاستجابة من Gemini.
  """
  # Create the model
  generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
  }

  model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-thinking-exp-1219",
    generation_config=generation_config,
  )

  # Upload the file to Gemini
  files = [
    upload_to_gemini(file_path, mime_type="image/png"),
  ]

  chat_session = model.start_chat(
    history=[
      {
        "role": "user",
        "parts": [
          files[0],
          "صف",
        ],
      }
    ]
  )

  response = chat_session.send_message("""أريدك أن تصف صورة ويب وصفًا تفصيليًا ودقيقًا بحيث يشمل كافة التفاصيل التي يحتاجها المبرمج لإعادة إنشاء الصورة بدقة. يجب أن يحتوي الوصف على ما يلي:**

    توزيع الأقسام في الصورة:
        عدد الأقسام الرئيسية والفرعية.
        موقع كل قسم بالنسبة للصفحة (على سبيل المثال: وسط الصفحة، الجزء العلوي، السفلي، الأيمن، الأيسر).
        إذا كان القسم داخل شكل (مثل مربع، دائرة، أو غيره)، أو إذا كان يتمركز بشكل حر.
        نسب وأبعاد كل قسم بالنسبة لباقي الصفحة.
        المسافات بين الأقسام والعناصر المختلفة.

    الألوان:
        لون خلفية كل قسم أو جزء من الصفحة.
        ألوان النصوص داخل كل قسم (النصوص الرئيسية والثانوية).
        ألوان الأيقونات أو الرموز أو الصور المرفقة.
        تدرجات الألوان إذا كانت موجودة (Gradients).
        وصف أي تأثيرات لونية، مثل الظلال (Shadows) أو التوهج (Glow).

    النصوص:
        استخراج النصوص المكتوبة بشكل دقيق.
        نوع الخط المستخدم (Font Family)، حجمه (Font Size)، سماكته (Font Weight)، ولونه (Color).
        تمركز النصوص (مثل: في الوسط، على اليمين، على اليسار، مائل، أو عمودي).
        إذا كان هناك نصوص مزخرفة أو تحتوي على تأثيرات (مثل خطوط تحتها، خطوط متقطعة، أو أي أنماط أخرى).

    الأيقونات والإيموجي:
        تحديد جميع الإيموجي أو الأيقونات الموجودة.
        وصف شكل الأيقونة (مثال: أيقونة على شكل سهم دائري، أيقونة على شكل منزل).
        مكان الأيقونات (داخل مربع، فوق النص، بجانب عنصر معين).
        ألوان الأيقونات وأحجامها النسبية.

    الخلفيات:
        وصف دقيق للخلفيات (إن كانت صورة، لون موحد، تدرج لوني، أو تصميم مخصص).
        وصف أي أنماط أو أشكال موجودة في الخلفية (مثل خطوط، نقاط، زخارف).
        وصف حواف الخلفيات (حادة، دائرية، أو بها تأثير).

    الأشكال الهندسية:
        تحديد جميع الأشكال الهندسية المستخدمة في التصميم (مثل مربعات، دوائر، مستطيلات، أو غيرها).
        وصف حجم كل شكل بالنسبة لبقية العناصر (النسب المئوية أو الوصف النسبي).
        ألوان وأبعاد الأشكال، وأي تفاصيل إضافية (مثل حواف مستديرة أو خطوط منقطة).

    الأقسام المميزة:
        وصف أماكن الأقسام المميزة في التصميم (مثل مكان العناوين، مكان الصور البارزة، الأقسام التفاعلية).
        هل الأقسام متمركزة (Centered) أم مثبتة على جانب معين.
        وجود خطوط فاصلة أو تقسيمات بين الأقسام.

    حجوم العناصر:
        ذكر الأحجام النسبية أو الدقيقة (بالنسبة للنوافذ أو العناصر القريبة).
        إذا كان هناك تكبير أو تصغير لأي عنصر معين.

    التفاعلات والعناصر الديناميكية:
        تحديد أماكن الأزرار والعناصر القابلة للتفاعل (مثل النقر أو السحب).
        وصف الأزرار (مثل لونها، شكلها، النص المكتوب عليها، وظيفتها).
        وصف أي تأثيرات عند التفاعل (Hover effects، أو تغيرات عند الضغط).

    صور ورموز أخرى:
        استخراج النصوص المكتوبة على الصور (OCR إذا أمكن).
        وصف الصور المستخدمة في الخلفيات أو داخل الأقسام (مثلاً: صورة لشجرة، صورة لأشخاص).
        أبعاد الصور وموقعها النسبي.""")
  return response

if __name__ == "__main__":
  # === START: Update file paths here ===
  file_path = "Capture.jpg" # <---- Replace this with your actual file path.
  # === END: Update file paths here ===

  response_object = get_gemini_response(file_path)
  print(response_object)

  # === START: Save response.text to file ===
  output_file = "response.md"  # اسم الملف لحفظ الاستجابة

  try:
      with open(output_file, "w", encoding="utf-8") as f:
          f.write(response_object.text)
  except Exception:
      pass  # تجاهل الأخطاء أثناء الحفظ لتبسيط العملية
  # === END: Save response.text to file ===