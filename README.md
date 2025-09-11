# Završni praktični projekt - Treniranje LLM (Large Language Model) mreže na užem problemu/domeni

**Naziv projekta:** PlotGen - Generiranje sinopsisa filmova  
**Opis projekta:** Treniranje GPT modela na datasetu koji sadrži sinopsise (plotove) filmova. Model nakon treniranja može generirati nove sinopsise.

# Struktura projekta

- `prepare.ipynb` — notebook za preprocesiranje raw podataka u binarni token niz
- `dataset/` — folder sa datasetovima
- `dataset/processed/train.bin`, `dataset/processed/val.bin` — izlaz tokenizacije, spremljeni kao uint16 memmap za brzo dohvaćanje batcheva
- `config.py` — glavni set hiperparametara modela i treniranja
- `train.py` — training loop (učitavanje data/processed datoteka i treniranje)
- `generate.py` — skripta za generaciju promptova
- `mingpt/` — implementacija GPT modela([minGPT](https://github.com/karpathy/minGPT))
- `requirements.txt` — Python paketi

# Postupak pokretanja

Postaviti virtualno okruženje i instalirati potrebne pakete iz `requirements.txt`.
U `/dataset` folder premjestiti dataset na kojem ce se trenirati model.
Pokrenuti `prepare.ipynb` za obradu i tokenizaciju dataseta. 
Zatim se moze pokrenuti treniranje modela uz `/train.py`.
U `train.py` implementirano je periodicno generiranje promptova tijekom treniranja radi praćenja napretka modela.
Da bi se pratili generirani promptovi, kao i loss modela, treba pokrenuti TensorBoard uz komandu:
```python
tensorboard --logdir runs --port 6006 --load_fast=false
```
Kada treniranje završi, uz `generate.py` gdje se učitaje checkpoint (`models/_nazivCheckpointa_.pt`) mogu se generirati promptovi.

# Dataset i preprocessing (prepare.ipynb)

Izvori datasetova:  
[CMU Movie Summary Corpus (cmu-plots)](https://www.cs.cmu.edu/~ark/personas/)   
[themoviespoiler.com (movieSpoiler-plots)](https://themoviespoiler.com/)

Linkovi za preuzimanje:  
[Dataset 1 - cmu-plots](https://drive.google.com/file/d/12PyNYAi1nrH07b-K0E4AKAt2A2sFh3ON/view?usp=drive_link)   
[Dataset 2 - movieSpoiler-plots](https://drive.google.com/file/d/1QiWSaRpE3wbtS8tdEnsrFDGyPC2s5wMT/view?usp=drive_link)

Dataset je tekstualna kolekcija plotova filmova. Da bi model mogao "čitati" ove plotove, potrebno je dati dataset preoblikovati tako što pretvaramo tekst u niz brojeva - tj. tokeniziramo tekst. Trivijalna tokenizacija bi se radila tako što bismo svaki karakter mapirali na jedinstveni broj. Radi bolje kvalitete modela - odlučio sam se na uporabu `tiktoken` tokenizatora koji koristi BPE (Byte Pair Encoding). BPE (Byte Pair Encoding) omogućava modelu da prepoznaje česte podriječi čime model tijekom treniranja više puta vidi iste tokene u različitim kontekstima što mu pomaže da bolje generalizira i razumije gramatička pravila jezika.

Upute:

```bash
# stvori i aktiviraj virtualenv
python -m venv plot-generator-env
source plot-generator-env/bin/activate
pip install -r requirements.txt

# u dataset folder premjesti jedan od datasetova na kojem ćeš trenirati model

# uz prepare.ipynb se vrsi tokenizacija teksta

#train.bin i val.bin su tip uint16 memmap datoteke koje omogućuju brzo učitavanje tj. uzimanje batcheva tijekom treniranja
```

Nakon toga stvaraju se `dataset/processed/train.bin` i `dataset/processed/val.bin`.

# Generiranje promptova: `generate.py`

- Učitava tokenizer (tiktoken) i model konfiguraciju.
- Učitava checkpoint (`models/checkpoint.pt` i stavlja model u `eval()` režim.
- Primjeri promptova su u listi `prompts`.
- Generacija se radi preko `model.generate(...)` s parametrima:
  - `max_new_tokens` (npr. 200) - maksimalni broj generiranih tokena
  - `temperature` (npr. 0.7) - kontrolira nasumičnost / kreativnost (nize vrijednosti - model uzima one najsigurnije tokene)
  - `do_sample=True` ili `False` - određuje hoće li se koristiti uzorkovanje ili ne (True = uzorkovanje, False = deterministički izlaz)
  - `top_k` (npr. 50) - broj najvjerojatnijih tokena koji se uzimaju u obzir pri generiranju

# Treniranja

## Specifikacije računala na kojem je treniran model:

**CPU:** AMD Ryzen 7 5700X3D 8-Core, 16-Thread  
**GPU:** [Palit RTX 4060 Ti StormX](https://www.techpowerup.com/gpu-specs/palit-rtx-4060-ti-stormx.b11180)  
**RAM:** G.SKILL RipjawsV DDR4 2x8GB

## 1. Treniranje

Sažetak toka:

- Učitao sam `train.bin` i `val.bin` kao numpy memmap.
- Warmup faza se koristi za postupno povećanje learning rate-a tijekom prvih `WARMUP_STEPS` iteracija jer doprinosi stabilnijem treniranju, to jest model neće u početku prebrzo učiti (mijenjati težine drastično). Nakon toga se koristi [cosine annealing](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html) do kraja treninga.
- Model se trenira koristeći [AdamW optimizator](https://www.datacamp.com/tutorial/adamw-optimizer-in-pytorch) s weight decay i betas (0.9, 0.95).
- Sekvence se uzimaju iz memmapa i šalju u model za treniranje.
- Gradijent se akumulira tijekom `GRADIENT_ACCUMULATION_STEPS` koraka. (Implementirano radi bolje efikasnosti memorije na GPU-u)
- Nakon akumulacije vrši se clipping gradijenta (`clip_grad_norm_`).
- Model se trenira koristeći mixed precision (AMP) i `torch.cuda.amp.GradScaler` za stabilno treniranje na GPU‑u.
- Svakih `EVAL_INTERVAL` iteracija poziva se `estimate_loss()` (prosjek `EVAL_ITERS` batcheva na train i val setu) i poziva se kratka generacija demo‑primjera.
- Na kraju treniranja sprema se checkpoint

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
## PROMPT: The movie begins with a soldier who is abandoned from his unit

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

Za prvi pokušaj treniranja modela, bio sam dosta zadovoljan jer sam dobivao donekle smislene plotove. Rečenice su gramatički većinom ispravne i imaju smisla. Po plotovima koje generira, vidi se da može ostati u stilu onoga što je u promptu bilo napisano (misterija, drama, komedija...). Ono čime nisam zadovoljan bio jest da priče i dalje djeluju malo nelogične, radnje često postaju besmislene i nemaju vezu s onime što je model početno generirao.

## 2. Treniranje

Dodao sam [TensorBoard](https://www.tensorflow.org/tensorboard) da bih imao bolji uvid u kretanje train i validation lossa, learning rate...

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
LEARNING_RATE = 1e-3
EVAL_ITERS = 50
MAX_GRAD_NORM = 1.0
WARMUP_STEPS = 500
```

![TensorBoard](https://i.imgur.com/BJXJFXw.png)

![Loss graph](https://i.imgur.com/BVcuQE5.png)
![Learning rate graph](https://i.imgur.com/bfANktT.png)

```markdown
## PROMPT: A little young boy manages to enter TV as portal

STORY: A little young boy manages to enter TV as portal
is being opened. The boy is then shown as a boy and
his mother, who tell the boy on the radio that he is
the son of the prince. The boy is then sent to the
city to be adopted by the prince and grows up to
be a wealthy man. His father, who was a king,
had abandoned his family when he was young.
The boy's father had been a servant to the prince,
ut the prince is ill and does not want to allow
him to return. The prince becomes a servant and
hen decides to send the boy to the palace.
he next day, the prince wakes up the next
morning to find the boy and the prince in
bed. The prince and the prince are shocked
and furious. The prince asks the prince
how he got the boy for the prince and the
boy runs home with the help of the prince
and the prince. The prince takes the boy
nd the prince to the palace. The prince tells
his mother that he will meet the prince.
```

Model sam uploadao na [Hugging Face](https://huggingface.co/to0ony/final-thesis-plotgen) te uz pomoć [Gradia](https://gradio.app/) napravio web sučelje za generiranje plotova - [PlotGenApp](https://huggingface.co/spaces/to0ony/final-thesis-plotgen-app).

![Aplikacija](https://i.imgur.com/rQY778d.png)
## 3. Treniranje

Model iz drugog treniranja sam uzeo i dotrenirao ga na drugom manjem datasetu. Koristio sam [movieSpoiler-plots](https://drive.google.com/file/d/1QiWSaRpE3wbtS8tdEnsrFDGyPC2s5wMT/view?usp=drive_link) dataset. Taj dataset je obogaceniji sa zavrsecima filmova pa sam time mislio postici da model nauci bolje povezivati radnju i da ima smislenije zavrsetke.

![Loss](https://i.imgur.com/k4LCH6R.png)

![LR](https://i.imgur.com/loF1fXM.png)


Primjer prompta generiranog nakon zavrsetka fine tunea (1600 step):

```markdown
PROMPT: Young man tries to escape forest when

Young man tries to escape forest when he is attacked by a monster. After they make it to the top and find the town, the monster tries to eat his dog, T'Chadwick. After throwing away their fire breath from the monster, they begin to flee. They get into the town and get inside a tunnel as it causes the town's residents to fall into the water. The villagers make it onto the walls to rescue the dog from the monster and use it to lure it closer to the roof of the town in hopes of climbing to the monster. Two months later, T'Chadwick is still dealing with the monster and its zombie apocalypse. While T'Chadwick looks on, other survivors see the monster's corpse before it attacks. After the monster becomes a large monster, the two run across the woods to run into the woods. The monster catches T'Chadwick while T'Chadwick runs through the woods and tries to help find a way into the town.
```
