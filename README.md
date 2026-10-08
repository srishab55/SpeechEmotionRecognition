# Adhoc Tests: CNN-BiLSTM on four emotion corpora, whole speakers held out

**Question.** The main study's best from-scratch network scored far below the wav2vec2 models. What does a
CNN-BiLSTM do when it reads frozen wav2vec2 frames and is trained on all four corpora (RAVDESS, CREMA-D, TESS,
SAVEE: 12,162 clips, 121 speakers), tested only on speakers it never heard?

**Protocol.** Identical to Part C of the extension study, so the numbers are paired with its SVM (0.777) and
logistic regression (0.736): `StratifiedGroupKFold(5)` over speakers, stratified on corpus x emotion, seeds 42,
43, 44. Inside each training set, about one sixth of the speakers become validation speakers for early stopping.
The held-out fold is predicted once. Scores are macro-F1 over the labels each set contains, pooled per seed and
averaged over seeds, with a speaker-level bootstrap interval. Per-corpus rows matter as much as the overall row:
CREMA-D is 61% of the clips, and corpus identity is a shortcut the model can partly use.

**Model.** `unheard_lib/nets.py::CNNLSTM(bidirectional=True)`, registered as `cnn_bilstm`: two Conv1d blocks
(kernel 5, channel-norm, ReLU, dropout, max-pool 2) then a bidirectional LSTM, masked mean over time, linear head.
Input: wav2vec2-base layer-6 frames (768-d), z-scored per speaker from the speaker's own audio (no labels).
Width 128, dropout 0.3, AdamW lr 1e-3, label smoothing 0.1, SpecAugment, batch 32, up to 40 epochs, patience 6.
Hyperparameters are fixed, not tuned (the main study tuned per fold); see Limits.

## Run it

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get the audio (not redistributed; each corpus keeps its own licence):

| Corpus | Put it here | Source |
|---|---|---|
| RAVDESS (1,440) | `data/data/Person01 ... Person24/*.wav` | https://zenodo.org/records/1188976, then `python scripts/prepare_data.py PATH/TO/unzipped` |
| CREMA-D (7,442) | `data_external/cremad/AudioWAV/*.wav` | `kaggle datasets download -d ejlok1/cremad --unzip -p data_external/cremad` |
| TESS (2,800) | `data_external/toronto-emotional-speech-set-tess/` | `kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess --unzip -p ...`; delete the nested duplicate folder `TESS Toronto emotional speech set data/TESS Toronto emotional speech set data` |
| SAVEE (480) | `data_external/surrey-audiovisual-expressed-emotion-savee/ALL/*.wav` | `kaggle datasets download -d ejlok1/surrey-audiovisual-expressed-emotion-savee --unzip -p ...` |

```bash
python run_all.py                   # index -> frames -> train -> report (resumable)
python run_all.py --stage report    # rescore from saved predictions
jupyter notebook cnn_bilstm_multicorpus.ipynb   # same steps, with charts
pytest -q                           # fast checks, no audio needed
```

The first run downloads `facebook/wav2vec2-base` (about 380 MB) from Hugging Face and needs internet.
Frame extraction encodes each clip alone (about 20 to 60 minutes on an Apple GPU), then writes a 6 GB
padded float16 cache to `features/` (plus 2.4 GB of raw frames that can be deleted afterwards).

## Outputs

`outputs/cnn_bilstm/`: `pred_seed*_fold*.npz` (held-out probabilities), `cnn_bilstm_partC_style.csv` (score per
set with interval), `cnn_bilstm_per_emotion_f1.csv`, `comparison_with_svm_logistic.csv`.

## Layout

```
run_all.py              the single entry point
multicorpus/            corpora.py (clip table), frames.py (wav2vec2 frames), train.py (folds, training, scoring)
unheard_lib/, ser/      the main study's network, training loop and audio loader, reused unchanged except for
                        the new bidirectional option in nets.py
reference/              the extension study's clip table and Part C scores (order check and comparison)
scripts/prepare_data.py arranges the RAVDESS zip into data/data/PersonNN
tests/                  synthetic-data tests of the model, folds and the train/score path
```

## Limits

Fixed hyperparameters; no per-fold search. Per-speaker normalisation assumes each speaker reads a balanced mix of
emotions (TESS yes; SAVEE and CREMA-D not exactly). This tests new speakers inside corpora already seen, not new
corpora. TESS (2 speakers) and SAVEE (4) get no interval.

