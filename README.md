# MonoASR

**MonoASR** is a frugal and unified multilingual automatic speech recognition (ASR) model that eliminates the need for language-specific adapters by leveraging a shared Universal Language Projection (ULP) module, guided by a learnable language token. This design makes the model lightweight, adaptable, and efficient.

Additionally, this approach enhances training stability and simplifies the integration of new languages without requiring full retraining, making it ideal for scalable multilingual ASR systems.

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
```

### Monolingual Training - MonoASR

```bash
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_kabyle.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_arabic.py
python Monolingual_Training/Monolingual\ MonoASR/MonoASR_french.py
```

### Progressive Training - Baseline

```bash
python Progressive_Training/Progressive\ Baseline/baseline_kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_arabic_+kabyle.py
python Progressive_Training/Progressive\ Baseline/baseline_french_+arabic_+kabyle.py
```

### Progressive Training - MonoASR

```bash
python Progressive_Training/Progressive\ MonoASR/MonoASR_kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_arabic_+kabyle.py
python Progressive_Training/Progressive\ MonoASR/MonoASR_french_+arabic_+kabyle.py
```

### Simultaneous Training - Baseline

```bash
python Simultaneous_Training/Simultaneous_Baseline.py
```

### Simultaneous Training - MonoASR

```bash
python Simultaneous_Training/Simultaneous_MonoASR.py
```
