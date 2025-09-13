import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from kawakura.main.run_all_parts import run_once
from sample_manager import SampleManager

# サンプルマネージャーの初期化
sample_manager = SampleManager()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Step 1: Analyze the uploaded image using run_all_parts.py
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    # prompt = request.form.get('prompt', '')  # Removed prompt functionality
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Google Cloud APIを使用
            # Run analysis using subprocess (same as command line)
            cmd = [
                sys.executable,
                '-m',
                'kawakura.main.run_all_parts',
                filepath
            ]
            
            # Note: prompt parameter is not used in current implementation
            # You may need to modify run_all_parts.py to accept prompt parameter
            
            # Execute the command
            app.logger.info(f"Running command: {' '.join(cmd)}")
            app.logger.info(f"Working directory: {Path(__file__).parent.parent}")
            
            # Check Google Cloud auth
            import os as os_module
            if 'GOOGLE_APPLICATION_CREDENTIALS' in os_module.environ:
                app.logger.info(f"GOOGLE_APPLICATION_CREDENTIALS: {os_module.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
            else:
                app.logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set")
            
            # Set up environment for Google Cloud auth
            env = os.environ.copy()
            
            # Check for credentials in order of preference:
            # 1. Service account (for deployment)
            service_account_path = Path(__file__).parent / 'credentials' / 'service-account.json'
            if service_account_path.exists():
                env['GOOGLE_APPLICATION_CREDENTIALS'] = str(service_account_path)
                app.logger.info(f"Using service account credentials from: {service_account_path}")
            elif 'GOOGLE_APPLICATION_CREDENTIALS' in env:
                # 2. Environment variable
                app.logger.info(f"Using credentials from environment: {env['GOOGLE_APPLICATION_CREDENTIALS']}")
            else:
                # 3. Default gcloud location
                gcloud_config = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
                if gcloud_config.exists():
                    env['GOOGLE_APPLICATION_CREDENTIALS'] = str(gcloud_config)
                    app.logger.info(f"Using gcloud credentials from: {gcloud_config}")
                else:
                    app.logger.warning("No Google Cloud credentials found!")
                    return jsonify({
                        'error': 'Google Cloud認証が必要です。認証を行ってください。',
                        'auth_required': True
                    }), 401
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent, env=env, timeout=300)
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else 'Analysis failed'
                app.logger.error(f"Analysis command failed: {' '.join(cmd)}")
                app.logger.error(f"stderr: {result.stderr}")
                app.logger.error(f"stdout: {result.stdout}")
                return jsonify({'error': error_msg}), 500
            
            # Parse JSON from stdout
            try:
                # Find JSON in stdout (it starts with '{' and ends with '}')
                stdout = result.stdout.strip()
                json_start = stdout.find('{')
                json_end = stdout.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = stdout[json_start:json_end]
                    results = json.loads(json_str)
                    app.logger.info("Successfully parsed JSON from stdout")
                else:
                    # If JSON not found in stdout, check outputs directory
                    outputs_dir = Path(__file__).parent.parent / 'outputs'
                    json_files = sorted(outputs_dir.glob('run_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    if json_files:
                        # Get the most recent file
                        with open(json_files[0], 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        app.logger.info(f"Loaded results from {json_files[0]}")
                    else:
                        return jsonify({'error': 'No analysis results found'}), 500
                        
            except json.JSONDecodeError as e:
                app.logger.error(f"Failed to parse JSON: {e}")
                app.logger.error(f"stdout content: {result.stdout}")
                return jsonify({'error': 'Failed to parse analysis results'}), 500
            
            # Update the input_image path in results to match the uploaded file location
            results['input_image'] = filepath
            
            # Copy to temp directory for easier access
            results_filename = f"analysis_{timestamp}.json"
            results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
            
            # Save the modified results
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'status': 'success',
                'results': results,
                'image_path': filepath,
                'results_path': results_path,
                'timestamp': timestamp
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            app.logger.error(f"Analysis error: {str(e)}")
            app.logger.error(f"Traceback: {error_details}")
            return jsonify({
                'error': str(e),
                'details': error_details if app.debug else None
            }), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/compose', methods=['POST'])
def compose_face():
    """
    Step 2: Compose face using face_parts_fitter.py with the analysis results
    """
    data = request.get_json()
    
    # デバッグ用ログ
    app.logger.info(f"Compose request data: {data}")
    
    if not data or 'results_path' not in data or 'image_path' not in data:
        app.logger.error(f"Missing parameters: data={data}")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    results_path = data['results_path']
    image_path = data['image_path']
    timestamp = data.get('timestamp', datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    app.logger.info(f"Compose paths: results={results_path}, image={image_path}")
    
    if not os.path.exists(results_path):
        app.logger.error(f"Results file not found: {results_path}")
        return jsonify({'error': f'Results file not found: {results_path}'}), 404
        
    if not os.path.exists(image_path):
        app.logger.error(f"Image file not found: {image_path}")
        return jsonify({'error': f'Image file not found: {image_path}'}), 404
    
    try:
        # Run face_parts_fitter.py as subprocess
        script_path = os.path.join(
            Path(__file__).parent.parent,
            'kawakura',
            'face_parts_fitter.py'
        )
        
        # Create output filename
        output_filename = f"composed_{timestamp}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # Execute the script
        cmd = [
            sys.executable,
            script_path,
            results_path,
            '--out', output_path
        ]
        
        app.logger.info(f"Executing command: {' '.join(cmd)}")
        
        # ワーキングディレクトリを明示的に設定
        project_root = Path(__file__).parent.parent
        app.logger.info(f"Working directory: {project_root}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        app.logger.info(f"Command stdout: {result.stdout}")
        app.logger.info(f"Command stderr: {result.stderr}")
        app.logger.info(f"Return code: {result.returncode}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else 'Face composition failed'
            app.logger.error(f"Subprocess failed: {error_msg}")
            return jsonify({'error': error_msg, 'returncode': result.returncode}), 500
        
        # Check if output file was created
        if not os.path.exists(output_path):
            app.logger.error(f"Output file not created: {output_path}")
            app.logger.error(f"Directory contents: {os.listdir(os.path.dirname(output_path)) if os.path.exists(os.path.dirname(output_path)) else 'Directory not found'}")
            return jsonify({'error': 'Composed image was not created', 'output_path': output_path}), 500
        
        return jsonify({
            'status': 'success',
            'composed_image': output_filename,
            'message': 'Face composition completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """
    Download composed image
    """
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/status/<task_id>')
def get_status(task_id):
    """
    Get processing status (for future async implementation)
    """
    # For now, return a simple status
    return jsonify({
        'status': 'completed',
        'task_id': task_id
    })

@app.route('/api/samples')
def get_samples():
    """
    サンプルデータの一覧を取得
    """
    samples = sample_manager.get_samples()
    return jsonify(samples)

@app.route('/compose-sample', methods=['POST'])
def compose_sample():
    """
    サンプルデータを使って顔を合成（認証不要）
    """
    data = request.get_json()
    
    if not data or 'sample_id' not in data:
        return jsonify({'error': 'Sample ID is required'}), 400
    
    sample = sample_manager.get_sample_by_id(data['sample_id'])
    if not sample:
        return jsonify({'error': 'Sample not found'}), 404
    
    # タイムスタンプ生成
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # サンプルに対応するJSONファイルのパスを取得
        sample_json_paths = {
            'sample1': 'outputs/run_20250830_165827.json',
            'sample2': 'outputs/run_20250830_164634.json'
        }
        
        if sample['id'] not in sample_json_paths:
            return jsonify({'error': 'Sample JSON not found'}), 404
        
        # 既存のJSONファイルのパス
        json_path = os.path.join(
            Path(__file__).parent.parent,
            sample_json_paths[sample['id']]
        )
        
        if not os.path.exists(json_path):
            return jsonify({'error': f'JSON file not found: {json_path}'}), 404
        
        # face_parts_fitter.pyを実行
        script_path = os.path.join(
            Path(__file__).parent.parent,
            'kawakura',
            'face_parts_fitter.py'
        )
        
        output_filename = f"composed_sample_{timestamp}.png"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        
        # 直接JSONファイルを渡す
        cmd = [
            sys.executable,
            str(script_path),
            str(json_path),
            '--out', output_path
        ]
        
        app.logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else 'Face composition failed'
            app.logger.error(f"Sample composition error: {error_msg}")
            app.logger.error(f"stdout: {result.stdout}")
            return jsonify({'error': error_msg}), 500
        
        # Check if output file was created
        if not os.path.exists(output_path):
            return jsonify({'error': 'Composed image was not created'}), 500
        
        return jsonify({
            'status': 'success',
            'composed_image': output_filename,
            'message': 'Face composition completed successfully'
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        app.logger.error(f"Sample compose error: {str(e)}")
        app.logger.error(f"Traceback: {error_details}")
        return jsonify({
            'error': str(e),
            'details': error_details if app.debug else None
        }), 500

@app.route('/api/auth-status')
def auth_status():
    """
    認証状態を確認
    """
    # Check if any form of authentication is available
    has_auth = False
    auth_type = None
    
    # Check service account
    service_account_path = Path(__file__).parent / 'credentials' / 'service-account.json'
    if service_account_path.exists():
        has_auth = True
        auth_type = 'service_account'
    elif 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        has_auth = True
        auth_type = 'environment'
    else:
        # Check default gcloud
        gcloud_config = Path.home() / '.config' / 'gcloud' / 'application_default_credentials.json'
        if gcloud_config.exists():
            has_auth = True
            auth_type = 'gcloud'
    
    return jsonify({
        'authenticated': has_auth,
        'auth_type': auth_type
    })

if __name__ == '__main__':
    # Development server
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 50)
    print("🔐 Face Analysis & Composition System")
    print("Server running at http://127.0.0.1:8081")
    print("=" * 50)
    
    app.run(debug=True, port=8081)
