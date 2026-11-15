# AI Sign Language Translator

A Django web application for live sign recognition, learning demonstrations, and
camera practice. The app includes three independently trained profiles:

- **ASL words (WLASL100):** 100 video-derived classes, with support for up to two hands.
- **ISL emergency signs:** 8 video-derived classes, evaluated with signer-independent splits.
- **ASL alphabet:** 26 static-image classes. The current model does not model motion for J/Z.

The trained inference weights and MediaPipe hand-landmarker asset are included.
Uploaded dataset archives, extracted landmark caches, local database contents,
and generated learning demos are not included.

## Run locally

Use Python 3.11 or 3.12 and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>.

For local development, Django uses a development-only key and debug mode by
default. Before deploying, set `DJANGO_SECRET_KEY` to a private random value,
set `DJANGO_DEBUG=false`, and set `DJANGO_ALLOWED_HOSTS` to a comma-separated
list of the actual host names. Production startup fails if the secret key is
missing.

## Dataset training

Dataset archives are not in this repository. Supply your own licensed datasets
when retraining:

```bash
python train_models.py --archive /path/to/wlasl100.zip
python train_isl_models.py --archive /path/to/isl-emergency.zip
python train_asl_alphabet.py --archive /path/to/asl-alphabet.zip
```

The scripts create landmark caches under `weights/`; these are ignored by Git.
The training scripts report their held-out metrics. Validation scores should
not be treated as a guarantee of live-camera performance. See
[DATASET_GUIDE.md](DATASET_GUIDE.md) for accepted formats, data collection
needs, evaluation guidance, and current recognition limits.

## Tests

```bash
python manage.py test translator_app --verbosity 1
```
