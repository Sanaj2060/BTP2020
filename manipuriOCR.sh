#!/bin/bash
echo "Initializing the Tesseract OCR for Manipuri Language"
echo "Please select the option to continue..."
echo "1. Manipuri-OCR (Bengali)"
echo "2. Transliteration (Bengali -> Meitei Mayek)"
echo "To quit press [crtl+c]"
read -p 'Enter option (1/2): ' opt
if [ $opt == 1 ]
then
	python src/ocr.py
elif [ $opt == 2 ]
then
	python src/translator.py
else
	echo "Please enter a valid option!!"
fi 
