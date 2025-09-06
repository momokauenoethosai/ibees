# Face Analysis Application Dependency Graph

## Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        UI Application                        │
│                    (Streamlit/Flask/etc)                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    run_all_parts.py                          │
│              (Feature Extraction & Search)                   │
├─────────────────────────────────────────────────────────────┤
│ Inputs: Image path, prompt                                  │
│ Outputs: JSON with selected face parts                      │
│ Process: Extract features → Embed text → Search BigQuery    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   face_parts_fitter.py                       │
│               (Face Landmark & Composition)                  │
├─────────────────────────────────────────────────────────────┤
│ Inputs: JSON results, original image                        │
│ Outputs: Composite face image                               │
│ Process: Detect landmarks → Place assets → Blend            │
└─────────────────────────────────────────────────────────────┘
```

## Detailed File Dependencies

### 1. run_all_parts.py Dependencies

```
run_all_parts.py
│
├── [Standard Library]
│   ├── sys
│   ├── json
│   ├── re
│   ├── pathlib
│   └── typing
│
├── [Local Modules]
│   ├── common_config.py
│   │   └── Configuration constants (MIN_SCORE, etc)
│   │
│   ├── utils_embed_bq.py
│   │   ├── Dependencies:
│   │   │   ├── numpy
│   │   │   ├── vertexai
│   │   │   └── google.cloud.bigquery
│   │   └── Function: search_by_text_in_category()
│   │
│   └── part_extractors/
│       ├── hair.py      ─┐
│       ├── eye.py       │
│       ├── eyebrow.py   │
│       ├── nose.py      │
│       ├── mouth.py     ├─── All use: utils_gemini.py
│       ├── ear.py       │              ├── vertexai
│       ├── outline.py   │              └── common_config.py
│       ├── acc.py       │
│       └── beard.py     ─┘
│
└── [External APIs]
    ├── Google Vertex AI (Gemini)
    └── Google BigQuery
```

### 2. face_parts_fitter.py Dependencies

```
face_parts_fitter.py
│
├── [Standard Library]
│   ├── pathlib
│   ├── math
│   ├── json
│   ├── time
│   └── argparse
│
├── [External Packages]
│   ├── cv2 (OpenCV)
│   │   └── Image processing operations
│   │
│   ├── mediapipe
│   │   └── Face mesh detection
│   │       ├── FACEMESH_NOSE
│   │       ├── FACEMESH_LIPS
│   │       ├── FACEMESH_LEFT_EYEBROW
│   │       ├── FACEMESH_RIGHT_EYEBROW
│   │       └── FACEMESH_FACE_OVAL
│   │
│   └── PIL (Pillow)
│       ├── Image
│       └── ImageOps
│
└── [No local dependencies - standalone]
```

## Data Flow

```
Input Image
    │
    ▼
┌─────────────────────┐
│  run_all_parts.py   │
└──────────┬──────────┘
           │
           ├──► Feature Extraction (Gemini AI)
           │    └── For each part (hair, eye, etc.)
           │
           ├──► Text Embedding (Vertex AI)
           │    └── Convert descriptions to vectors
           │
           ├──► BigQuery Search
           │    └── Find similar assets by vector
           │
           └──► JSON Output
                └── Selected parts with paths & scores
                    │
                    ▼
        ┌─────────────────────┐
        │ face_parts_fitter.py│
        └──────────┬──────────┘
                   │
                   ├──► MediaPipe Face Detection
                   │    └── Extract facial landmarks
                   │
                   ├──► Asset Loading
                   │    └── Load selected face parts
                   │
                   ├──► Coordinate Calculation
                   │    └── Position assets on landmarks
                   │
                   └──► Image Composition
                        └── Final composite image
```

## External Service Dependencies

### Google Cloud Services
- **Vertex AI**
  - Gemini model for feature extraction
  - Text embedding model for vector search
- **BigQuery**
  - Vector similarity search database
  - Asset metadata storage

### Python Package Dependencies

#### Currently in requirements.txt:
- `google-cloud-aiplatform==1.71.1`
- `google-cloud-bigquery==3.36.0`
- `vertexai==1.71.1`
- `numpy==1.26.4`
- `pillow==11.3.0`

#### Missing from requirements.txt:
- `opencv-python` (or `opencv-python-headless`)
- `mediapipe`

## Module Structure

```
kawakura/
├── main/
│   ├── run_all_parts.py     # Main analysis entry point
│   └── common_config.py     # Shared configuration
│
├── utils/
│   ├── utils_embed_bq.py    # BigQuery & embedding utilities
│   └── utils_gemini.py      # Gemini AI utilities
│
├── part_extractors/         # Feature extraction modules
│   ├── hair.py
│   ├── eye.py
│   ├── eyebrow.py
│   ├── nose.py
│   ├── mouth.py
│   ├── ear.py
│   ├── outline.py
│   ├── acc.py
│   └── beard.py
│
└── face_parts_fitter.py     # Face composition module
```

## Key Observations

1. **Modular Design**: Clear separation between feature extraction and composition
2. **Cloud-Native**: Heavy reliance on Google Cloud services
3. **AI-Powered**: Uses Gemini for understanding and describing facial features
4. **Vector Search**: Leverages BigQuery for similarity matching
5. **Computer Vision**: MediaPipe for accurate facial landmark detection