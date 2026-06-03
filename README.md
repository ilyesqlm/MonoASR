# MonoASR

**MonoASR** is a frugal and unified multilingual automatic speech recognition (ASR) model designed to efficiently support multiple languages within a single architecture, offering a lightweight and adaptable solution that reduces the complexity typically associated with multilingual systems while improving scalability; furthermore, it enhances training stability and enables the integration of new languages without requiring substantial architectural modifications or full retraining, making it particularly suitable for scalable multilingual ASR applications.

## 🧪 Training Strategies

- **Monolingual Training**: Each language is trained independently using its respective dataset.
- **Progressive Training**: Languages are introduced incrementally. The model is first trained on one language, then adapted to additional languages by updating shared parameters and language tokens—preserving prior knowledge while minimizing catastrophic forgetting.
- **Simultaneous Training**: All target languages are trained jointly from the beginning. This strategy promotes the development of universal representations but requires careful data balancing to avoid performance bias toward high-resource languages.

## 📊 Datasets

We use publicly available speech datasets hosted on Hugging Face:

| Language | Dataset Name                                                | Link                                                                                   |
|----------|-------------------------------------------------------------|----------------------------------------------------------------------------------------|
| Kabyle   | `TutlaytAI/kabyle_asr`                                      | [View Dataset](https://huggingface.co/datasets/TutlaytAI/kabyle_asr)                  |
| Arabic   | `Yahya-Mohamed/Arabic_Audio_Rev3_9643_2021_Dataset`         | [View Dataset](https://huggingface.co/datasets/Yahya-Mohamed/Arabic_Audio_Rev3_9643_2021_Dataset) |
| French   | `odunola/french-audio-preprocessed`                         | [View Dataset](https://huggingface.co/datasets/odunola/french-audio-preprocessed)     |
| Wolof    | `yigagilbert/alffa-wolof-asr-dataset-19hr`                  | [View Dataset](https://huggingface.co/datasets/yigagilbert/alffa-wolof-asr-dataset-19hr)     |
| Yoruba   | `Hidi-agili/yoruba_tts_dataset`                             | [View Dataset](https://huggingface.co/datasets/Hidi-agili/yoruba_tts_dataset)     |

These datasets are loaded using the 🤗 `datasets` library.

## ⚙️ Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📘 Usage

### Monolingual Training - Baseline

```bash
python Monolingual_Training/Monolingual\ Baseline/baseline_kabyle.py
python Monolingual_Training/Monolingual\ Baseline/baseline_arabic.py
python Monolingual_Training/Monolingual\ Baseline/baseline_french.py
python Monolingual_Training/Monolingual\ Baseline/baseline_wolof.py
python Monolingual_Training/Monolingual\ Baseline/baseline_yoruba.py
```

### Monolingual Training - MonoASR

```bash
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_kabyle.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_arabic.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_french.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_wolof.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_yoruba.py
```

### Progressive Training - Baseline

```bash
python Progressive_Training/Progressive\ Baseline/baseline_kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_arabic_+kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_french_+arabic_+kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_wolof_+french_+arabic_+kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_yoruba_+wolof_+french_+arabic_+kabyle.py
```

### Progressive Training - MonoASR

```bash
python Progressive_Training/Progressive\ MonoASR/MonoASR_kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_arabic_+kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_french_+arabic_+kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_wolof_+french_+arabic_+kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_yoruba_+wolof_+french_+arabic_+kabyle.py
```

### Simultaneous Training

```bash
python Simultaneous_Training/Simultaneous_MMS.py
python Simultaneous_Training/Simultaneous_Whisper_small.py
python Simultaneous_Training/Simultaneous_Whisper_medium.py
python Simultaneous_Training/Simultaneous_Whisper_large.py
python Simultaneous_Training/Simultaneous_MonoASR.py
```
