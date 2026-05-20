# ForensIQ — Image Forensics System

A web-based image forensics system that detects photo manipulation and AI-generated content using SIFT, SURF, AKAZE, ORB algorithms combined with CLIP-based deep learning.

## Features
- Copy-move forgery detection (SIFT, SURF, AKAZE, ORB, VB)
- Error Level Analysis (ELA)
- AI-generated image detection (CLIP + statistical signals)
- Heatmap visualization of manipulated regions
- Screenshot/digital content detection

## Requirements
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

## Usage
```bash
python app.py
```
Then open `http://localhost:5050` in your browser.

## CLIP Model Setup
Download `clip_processor.zip` via Google Colab (see docs) and extract to project root.

## Tech Stack
Python · Flask · OpenCV · PyTorch · CLIP · HTML/CSS/JS