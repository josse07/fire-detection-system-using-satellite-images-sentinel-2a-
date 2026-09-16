# AI Wildfire Detection & Risk Prediction System

A production-grade, end-to-end ML platform for detecting active wildfires and predicting wildfire risk using Sentinel-2 multi-spectral satellite imagery and deep learning architectures.

## Architecture

1. **Preprocessing**: Ingests Sentinel-2 `SAFE` archives, calculates geospatial indices (NDVI, NDWI, NBR, SAVI), and extracts tiled features.
2. **Modeling**:
    - **Baseline**: Random Forest classifier using pixel/tabular data for robust initial detection.
    - **Deep Learning**: ResNet-18 for image-level classification and U-Net for pixel-wise segmentation mapping of burn areas.
3. **Inference API**: A FastAPI backend that hosts the trained models for real-time inference.
4. **Dashboard**: A Streamlit interactive web application with Folium map integration for exploring risk heatmaps and performing on-the-fly inference.

## Project Structure

- `api/`: FastAPI server for live model inference.
- `dashboard/`: Streamlit interactive visualization app.
- `data/`: Raw and processed dataset storage.
- `deployment/`: Dockerfiles and infrastructure configuration.
- `feature_engineering/`: Feature extraction and geospatial data generation scripts.
- `models/`: PyTorch architecture definitions (U-Net, ResNet) and Scikit-learn Baseline.
- `preprocessing/`: Scripts for indices calculation, dataset normalization, and image loading.
- `training/`: Reusable training loops with logging, checkpointing, and validation.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Extraction**
   Run the extraction script to unpack Kaggle datasets and Sentinel-2 `.SAFE` archives:
   ```bash
   python extract.py
   ```

3. **Start the API Backend**
   ```bash
   uvicorn api.main:app --reload
   ```

4. **Start the Dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

## Docker Deployment
```bash
docker build -t wildfire-platform -f deployment/Dockerfile .
docker run -p 8000:8000 wildfire-platform
```
