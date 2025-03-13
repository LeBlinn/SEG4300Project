# Battery Type Classifier using a pre-trained CNN - SEG4300 Project
I am currently using resnet18 and getting promising results. \
My dataset is very small, so I am working on improving it \
(especially the NI-CD & NI-MH data)

## Deploy docker & Utilize
Run docker compose
```bash
cd docker
docker compose up
```
Go to website that is now up at 127.0.0.1:80

or

Use curl to try the api out
```bash
curl -X POST http://127.0.0.1:80/predict -F "image=@testimg.jpg"
```

## Steps to build/run model trainer
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Then run the Jupyter Notebook

## Acknowledgments
Parts of my dataset originate from various online datasets. \
These include:

[Singapore Battery Dataset](https://github.com/FriedrichZhao/Singapore_Battery_Dataset)

[Roboflow batteries Computer Vision Project](https://universe.roboflow.com/school-gchcr/batteries-1aib9)

[Google images](https://www.google.com/imghp?hl=en&authuser=0&ogbl)
