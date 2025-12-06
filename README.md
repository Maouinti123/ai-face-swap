# Face Swap AI

my attempt at building a face swap system from scratch. took way longer than expected but finally got it working decently.

swaps faces between images/videos while trying to keep expressions and lighting looking natural. not perfect but gets the job done for most cases.

## what it does

- swap faces in images (the main thing)
- video face swap with smoothing so it doesnt flicker like crazy
- webcam support if you want to mess around in real-time
- api server for integration with other stuff
- onnx export for faster inference

## how it works

basically a GAN setup with 3 networks:

1. **encoder** - resnet50 backbone, spits out 512-dim vector representing "who" the person is
2. **generator** - unet style with adain layers to inject the identity
3. **discriminator** - patchgan to keep things looking realistic

spent a lot of time tuning the loss functions to balance identity preservation vs image quality. still not perfect but way better than my first attempts.

## project structure

```
face-swap-ai/
├── api/                    # fastapi server
├── config/                 # settings and hyperparams
├── dataset/                # data loading, preprocessing
├── losses/                 # all the loss functions
├── models/                 # encoder, generator, discriminator
├── training/               # training loop + visualization
├── video/                  # video processing stuff
├── utils/                  # random helper functions
├── scripts/                # cli tools
├── tests/                  # unit tests
├── train.py                # main training script
└── Makefile                # useful commands
```

## setup

```bash
# the usual venv stuff
python -m venv venv
source venv/bin/activate

# install deps
pip install -r requirements.txt

# or just use make
make install
```

if you have a gpu (highly recommended, cpu is painfully slow):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## getting started

### 1. get the dataset

using LFW from kaggle. you'll need to set up kaggle api key first (google it, its pretty straightforward)

```bash
make download
# or: python train.py --download
```

### 2. train

```bash
make train
# or for a quick test: make train-quick
```

this takes a while. go grab coffee or something. checkpoints save to `checkpoints/`, loss graphs to `graphs/`

### 3. export for production

```bash
make export
```

converts to onnx which is like 2x faster for inference

### 4. run the api

```bash
make api
# server starts at localhost:8000
```

## using the api

once the server is running, you can hit these endpoints:

```bash
# check if its alive
curl http://localhost:8000/health

# swap faces (base64 encoded images)
curl -X POST http://localhost:8000/swap/image \
    -H "Content-Type: application/json" \
    -d '{"source_image": "data:image/jpeg;base64,...", "target_image": "..."}'

# or just upload files directly (easier imo)
curl -X POST http://localhost:8000/swap/image/upload \
    -F "source=@myface.jpg" \
    -F "target=@target.jpg" \
    --output result.jpg
```

theres also swagger docs at `http://localhost:8000/docs` if you prefer clicking around

## video stuff

```bash
# cli way
python scripts/process_video.py --source face.jpg --video input.mp4 --output out.mp4

# or with make
make swap-video SOURCE=face.jpg VIDEO=input.mp4
```

heads up: video processing is slow without gpu. like really slow. a 30 sec video might take 5+ mins on cpu

## training notes

the loss function is a mix of several things (took forever to balance these):

- **identity loss** - makes sure the output looks like the source person
- **reconstruction loss** - L1 + SSIM, keeps structure intact
- **adversarial loss** - the usual GAN stuff
- **perceptual loss** - vgg features so it doesnt look blurry
- **color loss** - matches skin tones

default hyperparams are in `config/settings.py`. the ones that matter most:

- batch size: 8 (lower if you run out of vram)
- lr for generator: 1e-4
- lr for discriminator: 4e-4 (yes its higher, helps with training stability)

## performance

tested on my rtx 3080:

| mode     | speed                            |
| -------- | -------------------------------- |
| pytorch  | ~45ms per image                  |
| onnx     | ~25ms                            |
| tensorrt | ~12ms (havent fully tested this) |

cpu is like 500ms+ so... yeah get a gpu

## known issues / limitations

stuff that doesnt work great yet:

- side profiles - works best with frontal faces
- glasses/occlusions - sometimes gets weird
- extreme lighting - can mess up skin tones
- LFW is a small dataset - results would probably be better with vggface2 or something bigger

## running tests

```bash
make test          # run all tests
make test-cov      # with coverage report
make verify        # quick sanity check
```

## todo

- [ ] better handling of side profiles
- [ ] face segmentation for cleaner blending
- [ ] maybe try a different architecture (stylegan based?)
- [ ] proper tensorrt support


## references

some stuff i found helpful while building this:

- insightface/arcface for the identity encoder idea
- the spade paper for normalization
- bunch of stackoverflow threads
