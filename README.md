# Satellite Land-Use Classifier

[![tests](https://github.com/eslammohibrahim/satellite-land-use-classifier/actions/workflows/tests.yml/badge.svg)](https://github.com/eslammohibrahim/satellite-land-use-classifier/actions/workflows/tests.yml)

Give it a satellite image and it tells you what's on the ground: forest, river, highway, crops, one of 10 classes. The model is in PyTorch and there's a small FastAPI service around it.

I fine-tuned a ResNet18 that was already trained on ImageNet, and on the 5,400 test images it gets 98.1%. The same network trained from zero for the same 8 epochs only got to 92.7%.

The more interesting part came after. I added noise to the test images and accuracy fell to 15%, not much better than picking one of the 10 classes at random. Retraining with noise, blur and lighting changes brought it back to 91.5%, and clean accuracy only lost 0.8 points. But it didn't help with JPEG compression, which I had left out of training on purpose. So it only becomes robust to the kinds of damage it was trained on.

![Example predictions](reports/figures/examples.png)

## Data

[EuroSAT RGB](https://huggingface.co/datasets/timm/eurosat-rgb) (MIT license): 27,000 Sentinel-2 satellite images, 64×64 pixels, 10 classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake.

I used the splits that come with the dataset (16,200 / 5,400 / 5,400) and made sure no image is in more than one of them, both by ID and by content. Some classes have more images than others, Pasture 1,195 in training against 1,863 for Residential, so next to accuracy I also report macro F1. The data isn't in the repo; `src/data.py` downloads it the first time.

## Approach

Every model got the same budget: 8 epochs, AdamW, a one-cycle learning rate schedule and label smoothing of 0.05, all on CPU. For augmentation I used random flips and 90° rotations, since a satellite image has no "up". Each model keeps the checkpoint with the best validation accuracy, and the test set is only used for the final numbers.

## Results

| Model | Parameters | Training time (CPU) | Validation acc | Test acc | Test macro F1 | Latency per image (CPU) |
|---|---|---|---|---|---|---|
| Small CNN, from scratch | 0.4M | 4.9 min | 93.8% | 94.5% | 0.944 | about 2 ms |
| ResNet18, from scratch | 11.2M | 8.7 min | 92.5% | 92.7% | 0.926 | about 9 ms |
| ResNet18, ImageNet pretrained | 11.2M | 8.6 min | 97.9% | 98.1% | 0.980 | about 9 ms |
| ResNet18 pretrained + robustness training | 11.2M | 12.7 min | 97.1% | 97.2% | 0.972 | about 9 ms |

![Training curves](reports/figures/training_curves.png)

Pretraining made the biggest difference. The pretrained ResNet18 was at 93.6% validation accuracy after one epoch, while the same network from scratch needed all 8 epochs to reach 92.5%. I didn't expect the small CNN to beat ResNet18 from scratch, but with only 8 epochs a model 28 times bigger doesn't have time to pay off. Both from-scratch curves were still going up at the end, so longer training would probably close the gap.

Per class (ResNet18 pretrained, test set):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | 0.972 | 0.987 | 0.979 |
| Forest | 0.992 | 0.990 | 0.991 |
| HerbaceousVegetation | 0.959 | 0.976 | 0.967 |
| Highway | 0.974 | 0.974 | 0.974 |
| Industrial | 0.988 | 0.986 | 0.987 |
| Pasture | 0.977 | 0.977 | 0.977 |
| PermanentCrop | 0.981 | 0.946 | 0.963 |
| Residential | 0.989 | 0.995 | 0.992 |
| River | 0.983 | 0.972 | 0.977 |
| SeaLake | 0.990 | 0.998 | 0.994 |

![Confusion matrix](reports/figures/confusion_matrix.png)

The mistakes it makes are understandable when you look at the images. PermanentCrop gets confused with HerbaceousVegetation (16 images) and AnnualCrop (10), because at 64×64 pixels an orchard or vineyard looks like any other green field. River gets called Highway 12 times, since both are a long thin line across the image.

## Robustness: finding and fixing a weakness

Real satellite images aren't always clean: there's sensor noise, haze, poor lighting, and compression. I tested every model on the test set under five corruptions:

| Corruption | Standard model | Robust model |
|---|---|---|
| None (clean) | 98.1% | 97.2% |
| Gaussian noise | 15.0% | 91.5% |
| Blur | 55.1% | 95.8% |
| Darker (60% brightness) | 70.2% | 96.3% |
| Low contrast (50%) | 75.3% | 95.7% |
| JPEG, quality 15 (*never seen in training*) | 55.6% | 59.3% |

![Robustness](reports/figures/robustness.png)

Noise hurt the most. I looked into why: EuroSAT images are low in contrast, and the pixel values inside a typical image only vary by about 0.095 on a 0 to 1 scale. The noise I tested with has a standard deviation of 0.08, so it's almost as strong as the image itself.

The robust model is the same pretrained ResNet18, trained with random noise, blur, brightness and contrast changes added to the training images. It's above 91% on every corruption it saw in training. I kept JPEG out of training to check whether the robustness would carry over to something new, and it mostly didn't (55.6% to 59.3%). Overall it gives up 0.8 points on clean images for about 76 points on noisy ones.

To decide which model the API should serve, I set a rule before looking at the test results: use the robust one if its validation accuracy is within 0.5 points of the standard one. It ended up 0.85 points behind, so the API serves the standard model by default. If your images are noisy or low quality, switching to the robust model is one environment variable (see below).

## API

The model is served with FastAPI. `/predict` accepts an image of any size (it's resized to 64×64) and returns the top 3 classes.

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@app/static/examples/residential.png"
```

```json
{"prediction":"Residential","confidence":0.9502,"top3":[{"label":"Residential","probability":0.9502},{"label":"SeaLake","probability":0.0069},{"label":"Highway","probability":0.0063}]}
```

| Endpoint | Description |
|---|---|
| `GET /` | Upload page with six example images to try |
| `POST /predict` | Classify an image: returns the prediction, confidence, and top 3 |
| `GET /health` | Service status |
| `GET /classes` | The 10 class names |
| `GET /docs` | Interactive API docs (Swagger) |

A file that isn't an image returns 400, and anything over 5 MB returns 413. You'll notice the confidence never goes much above 95 to 96%. That comes from the label smoothing during training, which stops the model from being overconfident.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate        # on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Serve the trained model (downloaded automatically from this repo's release on first start)
uvicorn app.main:app
# Then open http://127.0.0.1:8000

# Serve the robust model instead
$env:MODEL_PATH="models/resnet18_robust.pt"    # PowerShell; on macOS/Linux: export MODEL_PATH=models/resnet18_robust.pt
uvicorn app.main:app

# Retrain and re-evaluate everything (about 40 minutes on CPU)
python src/train.py --model small_cnn
python src/train.py --model resnet18_scratch
python src/train.py --model resnet18_pretrained
python src/train.py --model resnet18_robust
python src/evaluate.py

pytest                        # API and data tests
```

### With Docker

```bash
docker build -t land-use-api .
docker run -p 8000:8000 land-use-api
```

The image includes the trained weights and runs as a non-root user. On every push, CI builds the image, starts it, and checks that it classifies an example image correctly.

The trained weights for both ResNet18 models are attached to the [v1.0 release](https://github.com/eslammohibrahim/satellite-land-use-classifier/releases/tag/v1.0).

## Project structure

```
src/
  data.py        download, decoding, caching, augmentation and corruption
  models.py      small CNN and ResNet18 variants
  train.py       training loop with best-checkpoint selection
  evaluate.py    test metrics, robustness tests, figures
app/
  main.py        FastAPI service
  static/        upload page and example images
tests/
  test_api.py    endpoints, predictions, error handling
  test_data.py   augmentation and normalization
reports/
  figures/       charts used in this README
  training/      per-epoch training history for each model
  results.json   all test and robustness metrics
```

## What I'd do next

Add JPEG to the robustness training and test on other corruptions it hasn't seen, like haze and clouds. I'd also like to train for longer and try EfficientNet and ConvNeXt-Tiny.
