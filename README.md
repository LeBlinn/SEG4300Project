# Battery Type Classifier using a pre-trained CNN - SEG4300 Project
I am currently using resnet18 and getting promising results.
My dataset is very small, so I am working on improving it.

## Deploy docker & Utilize
Run docker compose
```bash
cd docker
docker compose up
```

Use curl to test function
```bash
curl -X POST http://127.0.0.1:5000/predict -F "image=@testimg.jpg"
```

## Steps to build
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Acknowledgments
Portions of my dataset come from different datasets online. \
These include:

[Singapore Battery Dataset](https://github.com/FriedrichZhao/Singapore_Battery_Dataset)

[Roboflow batteries Computer Vision Project](https://universe.roboflow.com/school-gchcr/batteries-1aib9)

[Google images](https://www.google.com/imghp?hl=en&authuser=0&ogbl)
