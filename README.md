# Treniranje LLM (Large Language Model) mreže na malom računalu na užem problemu/domeni #

Studenti trebaju odabrati temu za koju će neuronska mreža ili veliki jezični model generirati sadržaj.

## Uvod

Ovaj repozitorij sadrži pipeline za trening GPT‑stila jezičnog modela (minGPT) na užem domenskom skupu podataka — zbirci plotova filmova. Cilj je da model generira imaginarne plotove filmova. README daje tehnički pregled: priprema skupa podataka, tokenizacija, arhitektura modela i hiperparametri, detalji treninga (optimizer, scheduler, AMP, gradient accumulation) i način generacije pomoću istreniranog checkpointa.

## Struktura važnih datoteka
- `prepare.ipynb` — notebook za preprocesiranje raw podataka u binarni token niz
- `dataset/plot.txt` — (izvorni) tekst plotova
- `dataset/processed/train.bin`, `dataset/processed/val.bin` — izlaz tokenizacije, spremljeni kao uint16 memmap za brzo dohvaćanje batcheva
- `config.py` — glavni set hiperparametara modela i treniranja
- `train.py` — training loop (memmap input, AMP, grad accum, checkpointing)
- `generate.py` — skripta za generaciju promptova
- `mingpt/` — minimalna implementacija GPT modela([minGPT](https://github.com/karpathy/minGPT))
- `requirements.txt` — Python paketi

# Dataset i preprocessing (prepare.ipynb)

[Dataset 1 - Wikipedia plots](https://drive.google.com/file/d/12PyNYAi1nrH07b-K0E4AKAt2A2sFh3ON/view?usp=drive_link)

[Dataset 2 - moviespoiler-plots scrapped](https://drive.google.com/file/d/1QiWSaRpE3wbtS8tdEnsrFDGyPC2s5wMT/view?usp=drive_link)

Dataset je tekstualna kolekcija plotova filmova. Da bi model mogao "čitati" ove plotove, potrebno je dati dataset preoblikovati. Dataset preoblikujemo tako što pretvaramo tekst u niz brojeva - tj. tokeniziramo tekst. Trivijalna tokenizacija bi se radila tako što bismo svaki karakter mapirali na jedinstveni broj. Radi bolje kvalitete modela - odlučio sam se na uporabu `tiktoken` tokenizatora koji koristi BPE (Byte Pair Encoding). BPE (Byte Pair Encoding) omogućava modelu da prepoznaje česte podriječi čime model tijekom treniranja više puta vidi iste tokene u različitim kontekstimaa što mu pomaže da bolje generalizira i razumije gramatička pravila jezika.

Upute:

```bash
# stvori i aktiviraj virtualenv
python -m venv plot-generator-env
source plot-generator-env/bin/activate
pip install -r requirements.txt

# u dataset folder premjesti jedan od datasetova na kojem ćeš trenirati model

# uz prepare.ipynb ćeš tokenizirati tekst 

#train.bin i val.bin su tip uint16 memmap datoteke koje omogućuju brzo učitavanje tj. uzimanje batcheva tijekom treniranja
```

Nakon toga stvaraju se `dataset/processed/train.bin` i `dataset/processed/val.bin`.

# Treniranje `train.py`

Sažetak toka:
- Učitaje se `train.bin` i `val.bin` kao numpy memmap.
- Warmup faza se koristi za postupno povećanje learning rate-a tijekom prvih `WARMUP_STEPS` iteracija jer doprinosi stabilnijem treniranju, to jest model neće u početku prebrzo učiti (mijenjati težine drastično). Nakon toga se koristi [cosine annealing](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) do kraja treninga.
- Model se trenira koristeći [AdamW optimizator](https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch) s weight decay i betas (0.9, 0.95).
- Sekvence se uzimaju iz memmapa i šalju u model za treniranje.
- Gradijent se akumulira tijekom `GRADIENT_ACCUMULATION_STEPS` koraka. (Implementirao radi efikasnosti memorije na GPU-u)
- Nakon akumulacije vrši se clipping gradijenta (`clip_grad_norm_`) i optimizatorski korak.
- Model se trenira koristeći mixed precision (AMP) i `torch.cuda.amp.GradScaler` za stabilno treniranje na GPU‑u.
- Svakih `EVAL_INTERVAL` iteracija poziva se `estimate_loss()` (prosjek `EVAL_ITERS` batcheva na train i val setu) i poziva se kratka generacija demo‑primjera.
- Na kraju treniranja sprema se checkpoint:

# Generacija: `generate.py`

- Učitava tokenizer (tiktoken) i model konfiguraciju.
- Učitava checkpoint (`models/checkpoint.pt` i stavlja model u `eval()` režim.
- Primjeri promptova su u listi `prompts`.
- Generacija se radi preko `model.generate(...)` s parametrima:
	- `max_new_tokens` (npr. 200) - 
	- `temperature` (npr. 0.7) - 
	- `do_sample=True` ili `False` - 
	- `top_k` (npr. 50) - 

# Postupak pokretanja

Postaviti virtualno okruženje i instalirati potrebne pakete iz `requirements.txt`.
U datasetu premjestiti jedan od datasetova na kojem će se trenirati model.
Pokrenuti `prepare.ipynb` za tokenizaciju teksta.
Zatim pokrenuti `train.py` za treniranje modela.
Kada treniranje završi, pokrenuti `generate.py` gdje se učitava checkpoint (`models/checkpoint.pt`) i generiraju promptovi.

# Moja treniranja

**GPU : NVIDIA GeForce RTX 4060 Ti 8GB**

Prvo sam trenirao model na Wikipedia datasetu (72MB). Treniranje je trajalo 15 sati.
Koristio sam sljedeći config:

```yaml
# Model architecture hyperparameters
BLOCK_SIZE = 256
N_LAYER = 12
N_HEAD = 12
N_EMBD = 768
MODEL_TYPE = None

# Training hyperparameters
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
MAX_ITERS = 8000
EVAL_INTERVAL = 400
LEARNING_RATE = 3e-4
EVAL_ITERS = 50
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
```

Model je zauzeo oko 7.8 GB VRAM-a tijekom treniranja. Do 3000 iteracija, demo promptovi bili su dosta repetitivni. 

```markdown
**step 3600: train loss 3.7761, val loss 4.1895
**
A mysterious murder shocks the town when the townspeople are attacked by a group of vampires. The townspeople are led by the vampire, who are led by the vampire. The vampires are then attacked by the vampire, who kills the vampires. The vampires are then attacked by the vampire's vampire, who is then killed by the vampire....
```

Oko 5000-ite iteracije, promptovi su počeli pokazivati neke razumne slijedove u radnji. Manje je ponavljanja.

```markdown
**step 5600: train loss 3.4639, val loss 4.0296 
**
A mysterious murder shocks the town when a young woman is found murdered. The local police inspector is a police inspector who is also a close friend of the murdered girl. The murderer is the only witness to the murder. He is assigned to the case, but the sheriff is not convinced. He is told that the killer is actually the killer, and that the killer is actually the killer....
```

![Loss](https://i.imgur.com/EO2oe4e.png)

Primjer prompta generiranog nakon potpunog treninga:

```markdown
 PROMPT: The movie begins with a soldier who is abandoned from his unit
------------------------------------------------------------
 STORY:
...in a boat with a fisherman. The two make their way to
a small fishing village, which is home to
a woman who has a briefcase full of
money. The woman is still alive, but her
father is not there. After being sent to
a military hospital, he is rescued by a
woman who takes him to the hospital. She
asks him to give her the money she
needs to keep her in a safe place.
The woman turns out to be a woman,
who says she is in fact a vampire.
A soldier tells the woman that she is
a vampire and that she has been working
on a cure for her blood. The woman
agrees to let him go, but he refuses
to help her as she is not afraid
of the other vampires. He agrees and a
few years later, he is released. He wants
to return to the village to retrieve the
money. The woman is none other than the
old woman and she tells him to hide
her body in the old man's room.
```