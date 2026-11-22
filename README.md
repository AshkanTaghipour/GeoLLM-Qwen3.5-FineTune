# GeoLLM-Qwen3.5-FineTune

**نماذج لغوية متخصصة في جيولوجيا الاستكشاف والتنقيب عن المعادن، مبنية على عائلة Qwen 3.5.**

> هذا المستودع هو نسخة مخصصة لعمل نماذج GeoLLM على معالجات الكمبيوتر العادية (CPU) دون الحاجة إلى NVIDIA CUDA أو بطاقة كروت شاشة مناسبة للتعلم الآلي.

## التشغيل السريع على المعالج

هذا الطريق مناسب لحاسوب بمعالج مثل **Intel Core i5 الجيل السادس** وما فوق.

### المتطلبات

- نظام Windows أو Linux
- Python إصدار 3.10 أو 3.11
- ذاكرة RAM لا تقل عن:
  - 6 GB لموديل 0.8B
  - 10 GB لموديل 2B
  - نوصي بـ 16 GB أو أكثر عند استخدام موديلات أكبر
- قرص شاغر لا يقل عن 10 GB للموديل الأصغر
- اتصال إنترنت لتنزيل الملفات من Hugging Face

> لا يلزم وجود CUDA للتشغيل على المعالج. استخدم الموديل 0.8B أولًا، فهو الأصغر والأفضل للدوائر الضعيفة أو القديمة.

### 1. تنزيل المشروع

```bash
git clone --branch cpu https://github.com/abdulkiqh-oss/GeoLLM-Qwen3.5-FineTune.git
cd GeoLLM-Qwen3.5-FineTune
```

### 2. إنشاء بيئة Python معزولة

**Windows:**

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

**Linux:**

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

### 3. تثبيت إصدارات CPU فقط

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers accelerate safetensors huggingface_hub
```

لا تستخدم ملف `setup_env.sh` ولا تثبت حزم CUDA أو bitsandbytes عندما يكون هدفك التشغيل فقط على المعالج.

### 4. تشغيل النموذج

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --question "What geophysical methods can help target komatiite-hosted nickel sulphide deposits?"
```

لإجراء محادثة تفاعلية:

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

لتقليل زمن وتكلفة التوليد:

```bash
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --question "Summarize the key gold pathfinder elements." --max_new_tokens 128
```

## إعداد CPU في السكربت

استخدم نسخة `float32` للتوافق الأوسع مع المعالج، ولا تمرر `device_map="cuda"`. السكربت الأصلي يحتاج تعديل تحميل النموذج ليكون كالتالي:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "AshkanTaghipour/GeoLLM-Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
).to("cpu")
model.eval()

messages = [
    {"role": "system", "content": "You are a specialist geologist."},
    {"role": "user", "content": "What geophysical methods target nickel sulphide deposits?"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )

answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(answer)
```

## الجدول المقترح حسب الموارد

| الموديل | علميًا مناسب للـCPU؟ | RAM المقترحة | السرعة المتوقعة |
|---|---:|---:|---|
| 0.8B | نعم، الخيار الأول | 6–8 GB | بطيء، لكنه قابل للاستخدام |
| 2B | ممكن بذاكرة كافية | 10–12 GB | بطيء جدًا |
| 4B | فقط للأنظمة قوية | 16 GB+ | بطيء جدًا |
| 9B | غير مناسب لمعظم i5 قديمة | 24 GB+ | شديد البطء |
| 27B | غير عملي بمعالج بدون GPU | 64 GB+ | شبه غير قابل للاستخدام |

الأرقام تقريبية وقد ترتفع بسبب إصدار الـweights وإعدادات PyTorch وعدد خيوط المعالج.

## ما الذي لن يعمل على حاسوب i5 الجيل السادس؟

- `train.py` يحتاج NVIDIA GPU وCUDA ويعتمد على Unsloth وbitsandbytes؛ لا يعد التدريب على CPU خيارًا عمليًا.
- التعلم من الصفر أو Fine-tuning لن يكون عمليًا على المعالج وحده.
- موديلات 9B و27B تتطلب ذاكرة ووقتًا أكبر بكثير مما تملكه أجهزة المعالج القديمة.
- الأداء يعتمد كذلك على سرعة القرص؛ وجود SSD أفضل بكثير من HDD.

## خيارات أسرع

1. استخدم موديل 0.8B فقط.
2. استخدم الإكمال بحد أقصى 128–256 token.
3. نفّذ حديثًا قصيرًا، بدل تمرير محادثات طويلة.
4. استخدم كمية قليلة من خيوط المعالج إذا كنت تشاهد توقفًا كاملًا للنظام:

```bash
set OMP_NUM_THREADS=4
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

على Linux:

```bash
export OMP_NUM_THREADS=4
python inference.py --model AshkanTaghipour/GeoLLM-Qwen3.5-0.8B --interactive
```

## النماذج المتوفرة

| الموديل | Hugging Face | VRAM على GPU | مناسب للـCPU |
|---|---|---:|---|
| GeoLLM-Qwen3.5-0.8B | [AshkanTaghipour/GeoLLM-Qwen3.5-0.8B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-0.8B) | ~2 GB | نعم |
| GeoLLM-Qwen3.5-2B | [AshkanTaghipour/GeoLLM-Qwen3.5-2B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-2B) | ~5 GB | فقط ذاكرة مرتفعة |
| GeoLLM-Qwen3.5-4B | [AshkanTaghipour/GeoLLM-Qwen3.5-4B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-4B) | ~9 GB | غير مستحسن |
| GeoLLM-Qwen3.5-9B | [AshkanTaghipour/GeoLLM-Qwen3.5-9B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-9B) | ~19 GB | غير عملي |
| GeoLLM-Qwen3.5-27B | [AshkanTaghipour/GeoLLM-Qwen3.5-27B](https://huggingface.co/AshkanTaghipour/GeoLLM-Qwen3.5-27B) | ~55 GB | غير عملي |

## استخدام NVIDIA GPU اختياري

لا تحتاج CUDA للتشغيل المحلي على المعالج، لكن وجود GPU NVIDIA حديث يسرّع الاستدلال والتدريب بصورة كبيرة. لا يوصى بتثبيت حزم GPU على جهاز i5 إذا لم توجد بطاقة NVIDIA أو في حالة عدم استقرار تعريف CUDA.

## التدريب وإعادة الإنتاج

التدريب مصمم للـNVIDIA GPUs ويعتمد على بيئة Cluster وCUDA. لا يمثل هذا الفرع بديلاً لتدريب CPU؛ بل هدفه إبقاء الاستدلال قابلاً للتشغيل على أجهزة المعالجة فقط.

```bash
python prepare_data.py --input_dir ./training_splits_v2 --output_dir ./processed_data
python train.py --model_name unsloth/Qwen3.5-0.8B --epochs 5
python benchmark.py --epochs 5
python evaluate.py --model_dir ./finetuned_lora --run_name "cpu-check"
```

## بنية المشروع

```text
├── train.py                 # تدريب LoRA باعتماد CUDA/Unsloth
├── evaluate.py              # تقييم النموذج
├── benchmark.py             # مقارنة النماذج المختلفة
├── inference.py             # الاستدلال التفاعلي وطرح السؤال
├── prepare_data.py          # تجهيز البيانات للتدريب
├── model_registry.py        # قائمة عائلة نماذج Qwen 3.5
├── test_training.py          # اختبارات خط التدريب
├── test_benchmark.py         # اختبارات المقارنات
├── setup_env.sh             # إعداد بيئة GPU أصلية
├── requirements.txt         # متطلبات المشروع الأساسية
├── training_splits_v2/      # بيانات التدريب الخام
└── docs/
    ├── BENCHMARK.md
    └── EVALUATION.md
```

## التحقق من التثبيت

```bash
python -c "import torch, transformers; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Transformers:', transformers.__version__)"
```

على CPU يجب أن يطبع `CUDA: False`، وهذه حالة طبيعية.

## حقوق الاستخدام والترخيص

المستودع الأصلي مرخص تحت Apache-2.0، وأوزان النماذج fine-tuned تراعي ترخيص Qwen أيضًا. راجع ملفات الترخيص قبل إعادة التوزيع:

- [LICENSE الأصلي](LICENSE)
- [ترخيص وزن 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE)

المشروع الأصلي من إعداد AshkanTaghipour:

- GitHub: https://github.com/AshkanTaghipour
- Hugging Face: https://huggingface.co/AshkanTaghipour
