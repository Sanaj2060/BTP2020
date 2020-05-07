# Manipuri Language - Tesseract OCR and NMT

First objective is to train a OCR model to recognise Manipuri words using **[Tesseract](https://github.com/tesseract-ocr)** as backend engine.
Second is to develop a [transformer](https://arxiv.org/abs/1706.03762) model for **[NMT](https://en.wikipedia.org/wiki/Neural_machine_translation)** which will be used in transliteration from Bengali script to Meitei Mayek.
We also designed and develop the GUI feature for non-developers usage. 

## Steps to install
1. Clone the repository locally.
2. Install [Tesseract-4.x.x](https://tesseract-ocr.github.io/tessdoc/TrainingTesseract-4.00) version and its dependencies.
3. Move the *dataset/tesseract/**mni.traineddata*** to tessdata folder.
4. `pip install -r requirements.txt`
5. `./manipuriOCR.sh`
6. Happy Hacking
