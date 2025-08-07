**Treniranje LLM (Large Language Model) mreže na malom računalu na užem problemu/domeni**

Studenti trebaju odabrati temu za koju će neuronska mreža ili veliki jezični model generirati sadržaj.

**Zahtjevi projekta:**

1. Kreirati GitHub repozitorij
2. Prikupiti relevantne podatke
3. Odabrati ili izraditi vlastitu neuronsku mrežu/veliki jezični model (do 1B parametara)
4. Razviti skriptu za treniranje modela
5. Ocijeniti učinkovitost modela

**Pomoćni resursi:**

- [Video tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [GitHub repo - minGPT](https://github.com/karpathy/minGPT)
- [Hugging Face](https://huggingface.co/)
- [PyTorch](https://pytorch.org/)

# Instrukcije

## Kreiranje Virtual Environment

```bash
# Kreiraj novi environment
python -m venv plot-generator-env

# Aktiviraj environment (Windows)
plot-generator-env\Scripts\activate

# Activiraj environment (Linux/Mac)
source plot-generator-env/bin/activate

# Instaliraj dependencies
pip install -r requirements.txt
```

## Pokretanje Projekta

1. Aktiviraj environment
2. Pokreni data preprocessing: `jupyter notebook prepare.ipynb`
3. Pokreni treniranje: `python train.py`
4. Generiraj tekst: `python generate.py`

## Dataset

[Plotovi filmova](https://drive.google.com/file/d/1pPSv1I3qUXvgfBfIyO_Lh7FXuwUTeKQD/)
