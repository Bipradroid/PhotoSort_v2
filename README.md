# PhotoSort

AI-powered photo organiser — detects faces, embeds them with ArcFace, and clusters by person using **DBSCAN, HDBSCAN and Agglomerative Clustering**. User can choose which model to use from the dropdown button.

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python app.py
```

Then open **http://localhost:5000** in your browser.

## Usage

1. Zip your photos (JPG / JPEG / PNG) into a single `.zip` file.
2. Drop the ZIP onto the upload zone.
3. Wait while the app detects faces, generates ArcFace embeddings, and clusters.
4. Download the sorted folders — one folder per person, plus a `GroupPhotos` folder for images with 3+ faces.

## Pipeline

| Step | Tool |
|------|------|
| Face detection | RetinaFace |
| Face embedding | DeepFace / ArcFace (512-D) |
| Clustering | `sklearn.cluster.DBSCAN`(eps=0.55,min_samples=2,metric='cosine'), `HDBSCAN.hdbscan`(min_cluster_size=2,metric='euclidean',cluster_selection_method='eom') and `sklearn.cluster.AgglomerativeClustering` (cosine, average linkage, threshold = 0.67) |

## Notes

- Processing time scales with photo count and GPU availability. A CPU-only run on 100 photos takes ~5–15 min.
- Uploaded photos are stored under `jobs/` during processing and can be deleted afterward.
- Noise faces (very partial / blurry) are still assigned to their nearest cluster by Agglomerative Clustering, no `-1` noise label unlike DBSCAN.

## Why these models?

- **RetinaFace** was chosen for robust face detection under varying poses, lighting conditions, and occlusions.
- **ArcFace** generates highly discriminative 512-dimensional embeddings that work well for similarity-based clustering.

## Lessons from Version 1

The first version was made using **MTCNN** and **FaceNet** followed by clustering. It used a simpler pipeline and revealed several limitations:

- Partial faces sometimes produced noisy embeddings.
- Some images contained multiple people and required a separate GroupPhotos folder.
- DBSCAN occasionally assigned valid faces to the `-1` noise cluster.
- Agglomerative Clustering improved recall but can force uncertain faces into existing clusters.
- Different photo collections benefited from different clustering algorithms, which motivated the addition of multiple clustering modes.

## Future Improvements

- Confidence scores for cluster assignments 
- Better handling of side-profile faces
- GPU acceleration
- Web deployment
- Cluster preview thumbnails
