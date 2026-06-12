from flask import Flask, render_template, request, jsonify, send_file
import os, shutil, uuid, threading, zipfile
import numpy as np

print(os.getcwd())

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

jobs = {}  # job_id -> { status, progress, message, result }
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(BASE_DIR, "jobs")

def set_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    method = request.form.get('method', 'agglo')
    if not file or not file.filename.endswith('.zip'):
        return jsonify({'error': 'Please upload a .zip file.'}), 400

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    zip_path = os.path.join(job_dir, 'upload.zip')
    file.save(zip_path)

    jobs[job_id] = {
        'status': 'queued',
        'method': method,
        'progress': 0,
        'message': 'Queued…',
        'result': None,
        'job_dir' : job_dir
    }

    t = threading.Thread(target=process_job, args=(job_id, job_dir, zip_path, method), daemon=True)
    t.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    print(
        "STATUS:",
        job['status'],
        "PROGRESS:",
        job['progress']
    )
    return jsonify(job)



@app.route('/download/<job_id>')
def download(job_id):
    job = jobs.get(job_id)
    print("cwd =", os.getcwd())
    print("job_dir =", job['job_dir'])
    if not job or job['status'] != 'done':
        return jsonify({'error': 'Not ready'}), 400
    zip_path = job['zip_path']
    print(zip_path)
    print(os.path.exists(zip_path))
    return send_file(zip_path, as_attachment=True, download_name='sorted_photos.zip')


# ── Core processing ────────────────────────────────────────────────────────────

def process_job(job_id, job_dir, zip_path, method):
    try:
        # 1. Extract zip
        set_job(job_id, status='processing', progress=5, message='Extracting photos…')
        input_dir = os.path.join(job_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(input_dir)

        image_exts = ('.jpg', '.jpeg', '.png')
        image_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(input_dir)
            for f in files
            if f.lower().endswith(image_exts)
        ]

        if not image_files:
            set_job(job_id, status='error', message='No JPG or PNG images found inside the ZIP.')
            return

        # 2. Detect faces
        set_job(job_id, progress=12, message='Loading face detector…')
        from retinaface import RetinaFace
        from PIL import Image

        faces_dir = os.path.join(job_dir, 'faces')
        os.makedirs(faces_dir, exist_ok=True)

        face_to_image = {}
        image_face_count = {}
        total = len(image_files)

        for idx, path in enumerate(image_files):
            pct = 15 + int((idx / total) * 30)
            set_job(job_id, progress=pct, message=f'Scanning photo {idx + 1} of {total}…')
            try:
                img = Image.open(path).convert('RGB')
                pixels = np.array(img)
                faces = RetinaFace.detect_faces(pixels)
                if not isinstance(faces, dict):
                    continue
                image_face_count[path] = len(faces)
                for i, key in enumerate(faces):
                    x1, y1, x2, y2 = faces[key]['facial_area']
                    x1, y1 = max(0, x1), max(0, y1)
                    crop = pixels[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_img = Image.fromarray(crop).resize((160, 160))
                    name = f"{os.path.splitext(os.path.basename(path))[0]}_{i}.jpg"
                    face_to_image[name] = path
                    crop_img.save(os.path.join(faces_dir, name))
            except Exception:
                continue

        face_files = sorted(os.listdir(faces_dir))
        if not face_files:
            set_job(job_id, status='error', message='No faces detected in the uploaded photos.')
            return

        # 3. Generate embeddings
        set_job(job_id, progress=46, message=f'Encoding {len(face_files)} faces with ArcFace…')
        from deepface import DeepFace
        from sklearn.preprocessing import normalize

        embeddings, valid_faces = [], []
        for idx, ff in enumerate(face_files):
            pct = 46 + int((idx / len(face_files)) * 28)
            set_job(job_id, progress=pct, message=f'Encoding face {idx + 1} of {len(face_files)}…')
            try:
                emb = DeepFace.represent(
                    img_path=os.path.join(faces_dir, ff),
                    model_name='ArcFace',
                    detector_backend='skip',
                )[0]['embedding']
                embeddings.append(emb)
                valid_faces.append(ff)
            except Exception:
                continue

        if len(embeddings) < 2:
            set_job(job_id, status='error', message='Not enough faces to cluster (need at least 2).')
            return

        embeddings = normalize(embeddings)

        # 4. clustering
        set_job(job_id, progress=76, message='Grouping faces with Agglomerative Clustering…')
        

        if method == "dbscan":

            from sklearn.cluster import DBSCAN

            clusterer = DBSCAN(
                eps=0.55,
                min_samples=2,
                metric='cosine'
            )
        elif method == "hdbscan":

            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                metric='euclidean',
                cluster_selection_method='eom'
            )
        elif method == "agglo":

            from sklearn.cluster import AgglomerativeClustering

            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.67,
                metric='cosine',
                linkage='average'
            )

        labels = clusterer.fit_predict(embeddings)

        # 5. Organise into folders
        set_job(job_id, progress=85, message='Organising photos by person…')
        clusters_dir = os.path.join(job_dir, 'clusters')
        os.makedirs(clusters_dir, exist_ok=True)

        cluster_info = {}
        for face_file, label in zip(valid_faces, labels):
            orig = face_to_image.get(face_file)
            if not orig:
                continue
            folder_name = f'Person_{int(label) + 1:02d}'
            dest_dir = os.path.join(clusters_dir, folder_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, os.path.basename(orig))
            if not os.path.exists(dest):
                shutil.copy(orig, dest)
            cluster_info.setdefault(folder_name, [])
            base = os.path.basename(orig)
            if base not in cluster_info[folder_name]:
                cluster_info[folder_name].append(base)

        # 6. Group photos (≥3 faces)
        set_job(job_id, progress=92, message='Finding group photos…')
        group_dir = os.path.join(clusters_dir, 'GroupPhotos')
        os.makedirs(group_dir, exist_ok=True)
        group_photos = []
        for img_path, count in image_face_count.items():
            if count >= 3:
                dest = os.path.join(group_dir, os.path.basename(img_path))
                if not os.path.exists(dest):
                    shutil.copy(img_path, dest)
                group_photos.append(os.path.basename(img_path))

        if group_photos:
            cluster_info['GroupPhotos'] = group_photos

        # 7. Zip results
        set_job(job_id, progress=97, message='Packing download…')
        out_base = os.path.join(job_dir, 'result')
        shutil.make_archive(out_base, 'zip', clusters_dir)
        jobs[job_id]['zip_path'] = os.path.join(job_dir, 'result.zip')

        # Sort cluster_info so people come before GroupPhotos
        sorted_info = dict(sorted(cluster_info.items(), key=lambda x: (x[0] == 'GroupPhotos', x[0])))

        jobs[job_id].update({
            'status': 'done',
            'progress': 100,
            'message': 'Done!',
            'result': {
                'num_photos': total,
                'num_faces': len(valid_faces),
                'num_clusters': len(set(labels)),
                'cluster_info': sorted_info,
            },
        })

    except Exception as exc:
        set_job(job_id, status='error', message=f'Error: {exc}')


if __name__ == '__main__':
    os.makedirs(JOBS_DIR, exist_ok=True)
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=7860)
