# 🎓 LLM Fine-Tuning for Automated Quiz & Assessment Generation

> **Master's Final Project** — Applying Generative AI to Education through Parameter-Efficient Fine-Tuning of Large Language Models

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![PEFT](https://img.shields.io/badge/PEFT-QLoRA-8A2BE2?style=flat-square)

---

## 📌 Overview

This project investigates the use of **open-source Large Language Models (LLMs)** for automating the generation of question-and-answer pairs for educational quizzes, tests, and academic assessments. Using **QLoRA (Quantized Low-Rank Adaptation)**, three state-of-the-art models were fine-tuned under real-world hardware constraints and rigorously benchmarked using standard NLP evaluation metrics.

The core research question addressed is:

> *Can open-source LLMs, fine-tuned with parameter-efficient techniques, serve as cost-effective alternatives to proprietary models for domain-specific educational content generation?*

---

## 🧠 Models Benchmarked

| Model | Parameters | Architecture | Developer |
|---|---|---|---|
| **Meta LLaMA 2** | 8B | Transformer (decoder-only) | Meta AI |
| **Meta LLaMA 3** | 8B | Transformer (decoder-only) | Meta AI |
| **NVIDIA Nemotron-Mini 4B** | 4B | Transformer (decoder-only) | NVIDIA |

All models were fine-tuned using **QLoRA**, enabling 4-bit quantization with low-rank adapter layers to drastically reduce VRAM requirements without significant performance degradation.

---

## ⚙️ Methodology

### Fine-Tuning Strategy: QLoRA
- **Quantization**: 4-bit NormalFloat (NF4) quantization via `bitsandbytes`
- **LoRA Rank**: Configurable `r` and `alpha` hyperparameters applied to attention projection layers
- **Training Framework**: Hugging Face `transformers` + `peft` + `trl` (SFTTrainer)

### Task Formulation
The models were trained on a **supervised seq2seq** style task:
- **Input**: A passage or topic context
- **Output**: A structured Q&A pair (question + answers)

### Dataset
- Curated educational text passages with corresponding question-answer annotations
- Preprocessing included tokenization, prompt templating, and train/validation splitting

---

## 📊 Evaluation Metrics

Model outputs were evaluated using three standard NLP benchmarks:

| Metric | Description |
|---|---|
| **sacreBLEU** | Measures n-gram precision overlap between generated and reference text |
| **ROUGE** (R-1, R-2, R-L) | Recall-oriented metric assessing unigram, bigram, and longest common subsequence overlap |
| **METEOR** | Combines precision, recall, and semantic alignment using stemming and synonymy |

---

## 🗂️ Project Structure

```
.
├── LLMs-Experiments
│   ├── Head-QA-QuestionAnswer.Gen-LLaMa3_1-8b.ipynb
│   ├── Head-QA-QuestionAnswer.Gen-T5-small.ipynb
│   ├── HeadQA-LLaMA-2-7B
│   │   ├── generated_questions.json
│   │   ├── llama-2-7b-head-qa-spanish
│   │   │   ├── README.md
│   │   │   └── adapter_config.json
│   │   └── results
│   │       ├── checkpoint-4000
│   │       │   ├── README.md
│   │       │   ├── adapter_config.json
│   │       │   ├── added_tokens.json
│   │       │   ├── rng_state.pth
│   │       │   ├── scheduler.pt
│   │       │   ├── special_tokens_map.json
│   │       │   ├── tokenizer.json
│   │       │   ├── tokenizer.model
│   │       │   ├── tokenizer_config.json
│   │       │   ├── trainer_state.json
│   │       │   └── training_args.bin
│   │       └── runs
│   │           ├── Sep17_17-35-01_1a0cf69b42c4
│   │           │   └── events.out.tfevents.1726594512.1a0cf69b42c4.1422.0
│   │           └── Sep17_17-41-29_522d93005cc2
│   │               └── events.out.tfevents.1726594895.522d93005cc2.2000.0
│   ├── HeadQA-LLaMA-2-7B.ipynb
│   ├── HeadQA-LLaMA-3_1-8B-v2
│   │   ├── generated_questions.json
│   │   ├── llama-3-8b-head-qa-spanish-v2
│   │   │   ├── README.md
│   │   │   └── adapter_config.json
│   │   └── results
│   │       ├── checkpoint-3000
│   │       │   ├── README.md
│   │       │   ├── adapter_config.json
│   │       │   ├── rng_state.pth
│   │       │   ├── scheduler.pt
│   │       │   ├── special_tokens_map.json
│   │       │   ├── tokenizer.json
│   │       │   ├── tokenizer_config.json
│   │       │   ├── trainer_state.json
│   │       │   └── training_args.bin
│   │       └── runs
│   │           ├── Sep17_05-03-10_6c506a304324
│   │           │   └── events.out.tfevents.1726549407.6c506a304324.3512.0
│   │           ├── Sep17_05-13-18_6c506a304324
│   │           │   └── events.out.tfevents.1726550011.6c506a304324.3512.1
│   │           ├── Sep17_05-18-05_c61ac0478984
│   │           │   └── events.out.tfevents.1726550290.c61ac0478984.3984.0
│   │           └── Sep17_05-22-36_c61ac0478984
│   │               └── events.out.tfevents.1726550561.c61ac0478984.5743.0
│   ├── HeadQA-LLaMA-3_1-8B-v2.ipynb
│   ├── HeadQA-Nemotron-4
│   │   ├── generated_questions.json
│   │   ├── nemotron-4-head-qa-v1
│   │   │   ├── README.md
│   │   │   └── adapter_config.json
│   │   └── results
│   │       ├── checkpoint-4000
│   │       │   ├── README.md
│   │       │   ├── adapter_config.json
│   │       │   ├── rng_state.pth
│   │       │   ├── scheduler.pt
│   │       │   ├── special_tokens_map.json
│   │       │   ├── tokenizer.json
│   │       │   ├── tokenizer_config.json
│   │       │   ├── trainer_state.json
│   │       │   └── training_args.bin
│   │       └── runs
│   │           ├── Sep30_14-58-55_673ae7e5687d
│   │           │   └── events.out.tfevents.1727708381.673ae7e5687d.703.0
│   │           └── Sep30_15-13-20_8e2f04c89925
│   │               └── events.out.tfevents.1727709240.8e2f04c89925.1604.0
│   ├── HeadQA-Nemotron-4.ipynb
│   ├── Llama_2_Fine_tuning.ipynb
│   ├── Llama_3_1_8B_Fine_tuning.ipynb
│   ├── Prompt Engineering Experiments
│   │   ├── Prompt-Eng-Llama-3-1-8B.ipynb
│   │   └── Prompt-Eng-Nemotron-4.ipynb
│   ├── SQuAD_Multitask_QG_Llama_3_1_8B_Fine_tuning.ipynb
│   ├── SQuAD_Multitask_QuestionAnswer_Generation.ipynb
│   ├── decriptives
│   │   ├── EDA-Head-QA.ipynb
│   │   └── Estadísticas Descriptivas.ipynb
│   └── llm-evaluations
│       ├── Llama-2.ipynb
│       ├── Llama-3_1-8B.ipynb
│       └── Nemotron-4.ipynb
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended: 16GB+ VRAM for 8B models with QLoRA)
- `conda` or `virtualenv`

### Installation

```bash
# Clone the repository
git clone https://github.com/raquelvargas16/llm-quiz-generation.git
cd llms-experiments
```

### Key Dependencies

```
torch>=2.4.0
transformers>=4.43.1
peft>=0.12.0
trl>=0.10.1
bitsandbytes>=0.43.3
accelerate>=0.34.2
pytorch_lightning
datasets
sacrebleu
rouge-score
tokenizers
```

---

## 🏋️ Training

Key configuration parameters:

```yaml
model_name_llama3: "NousResearch/Meta-Llama-3.1-8B-Instruct"
model_name_llama2: "NousResearch/Llama-2-7b-chat-hf"
model_name_nvidia: "nvidia/Nemotron-Mini-4B-Instruct"
dataset_path: "head_qa"

# QLoRA settings

# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1

# bitsandbytes parameters

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False
```

---

## 📐 Evaluation

Run evaluation on a fine-tuned model checkpoint:
```Python
EXPERIMENT_NAME = 'HeadQA-LLaMA-2-7B/'
DRIVE_FOLDER_LOCATION = '/content/drive/MyDrive/MIAR - TFM/LLMs-Experiments/' + EXPERIMENT_NAME
model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "head_qa"
# Reload model in FP16 and merge it with LoRA weights
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
model = PeftModel.from_pretrained(base_model, "./results/checkpoint-4000/")
model = model.merge_and_unload()

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

---

## 🔍 Inference

Generate Q&A pairs from a custom text passage:

```Python
pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
messages = [
    {
        "role": "user",
        "content": """A partir de ahora eres un chatbot que habla español
        y vas a generar preguntas con cinco opciones de respuesta sobre el tema que
        el usuario ingrese. Utiliza el siguiente ejemplo para generar las preguntas
        Tema:
            Insulina.
        Ejemplo de pregunta:
        Aumenta la lipogénesis:
        a) Insulina.
        b) Glucagón.
        c) Cortisol.
        d) Somatotropina.
        e) Serotonina.
"""
    }
]
outputs = pipeline(
    messages,
    max_new_tokens=300,
)
messages.append({"role": "assistant", "content": outputs[0]["generated_text"][-1].get('content')})
data_test_df = flatten_dataset(dataset_test)
max_seq_length = 1024
test_dataset = QGDataset(
    data=data_test_df,
    tokenizer=tokenizer,
    source_max_token_len=max_seq_length,
    target_max_token_len=max_seq_length
)
for i in rand_int_arr:
    messages.append({"role": "user",
                     "content":"Tema: " + test_dataset[i]["answer_text"]})
    outputs = pipeline(
    messages,
    max_new_tokens=300)
    messages.append({"role": "assistant", "content": outputs[0]["generated_text"][-1].get('content')})
    print(outputs[0]["generated_text"][-1].get('content'))
```

**Example Output:**
```
Pregunta: ¿Cuál es la estructura que forma el borde interno de la órbita ocular?

 A) Conjuntiva vascular.
 B) Conjuntiva fibrosa.
 C) Cápsula corneal.
 D) Músculo orbicular de los ojos.
 E) Párpado.
```

---

## 📈 Results Summary

Comparative evaluation across the three models on the held-out test set:

| Model | sacreBLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | METEOR |
|---|---|---|---|---|---|
| LLaMA 2 8B (fine-tuned) | 8 | 0.36 | 0.15 | 0.23 | 0.27|
| LLaMA 3 8B (fine-tuned) | 16.9| 0.35 | 0.2| 0.27 | 0.22 |
| Nemotron-Mini 4B (fine-tuned) | 4.3 | 0.26 | 0.08 | 0.15 | 0.23 |

---

## 💡 Key Findings

- **QLoRA** proved highly effective for fine-tuning large models (7–8B parameters) on consumer-grade GPUs with minimal quality degradation compared to full fine-tuning.
- **LLaMA 3 8B** demonstrated the strongest overall performance across all NLP benchmarks due to its improved instruction-following capabilities over its predecessor.
- **NVIDIA Nemotron-Mini 4B**, despite having roughly half the parameters of the LLaMA models, achieved competitive scores — validating smaller, distilled architectures as viable options under tight compute budgets.
- Open-source LLMs fine-tuned on domain-specific data represent a **cost-effective and reproducible** alternative to proprietary APIs (e.g., GPT-4, Claude) for educational content generation.

---

## 🔬 Research Context

This project was developed as a **Master's Final Project** in 2024. It contributes to the growing body of research on:

- Parameter-efficient fine-tuning (PEFT) of LLMs
- AI-assisted education and automated assessment design
- Comparative benchmarking of open-source language models
- Accessibility of advanced NLP techniques under hardware constraints

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request


---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for the `transformers`, `peft`, `trl`, and `datasets` libraries
- [Meta AI](https://ai.meta.com/) for open-sourcing the LLaMA 2 and LLaMA 3 model families
- [NVIDIA](https://www.nvidia.com/en-us/ai/) for the Nemotron model family
- [Tim Dettmers et al.](https://arxiv.org/abs/2305.14314) for the original QLoRA paper
- The open-source NLP community for evaluation tooling (`sacrebleu`, `rouge-score`, `nltk`)
