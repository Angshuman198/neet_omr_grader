import numpy as np
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import warnings
import sys

# Import your existing OMR code (assuming it's in omr_processing.py)
# You'll need to save your existing code in a file called omr_processing.py
try:
    from omr_score_rank import lam
except ImportError:
    print("Error: Please save your OMR code in a file called 'omr_processing.py'")
    sys.exit(1)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Suppress warnings
warnings.filterwarnings("ignore")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    correct_path = None
    student_path = None

    try:
        # Check if files were uploaded
        if 'correct_answers' not in request.files or 'student_answers' not in request.files:
            return jsonify({'error': 'Both answer key and student answer sheet are required'}), 400

        correct_file = request.files['correct_answers']
        student_file = request.files['student_answers']

        # Check if files are selected
        if correct_file.filename == '' or student_file.filename == '':
            return jsonify({'error': 'Please select both files'}), 400

        # Check file extensions
        if not (allowed_file(correct_file.filename) and allowed_file(student_file.filename)):
            return jsonify({'error': 'Only image files (PNG, JPG, JPEG, BMP, TIFF) are allowed'}), 400

        # Save uploaded files
        correct_filename = secure_filename(correct_file.filename)
        student_filename = secure_filename(student_file.filename)

        # Add timestamp to avoid filename conflicts
        import time
        timestamp = str(int(time.time()))
        correct_filename = f"{timestamp}_correct_{correct_filename}"
        student_filename = f"{timestamp}_student_{student_filename}"

        correct_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], correct_filename))
        student_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], student_filename))

        correct_file.save(correct_path)
        student_file.save(student_path)

        print(f"Files saved: {correct_path}, {student_path}")

        # Verify files exist and are readable
        if not os.path.exists(correct_path) or not os.path.exists(student_path):
            return jsonify({'error': 'Failed to save uploaded files'}), 500

        print(
            f"File sizes - Correct: {os.path.getsize(correct_path)} bytes, Student: {os.path.getsize(student_path)} bytes")

        # Process the images using your lam function
        try:
            print("Calling lam function...")

            # Import and call your lam function
            from omr_score_rank import lam

            # Call with absolute paths
            marks, rank = lam(correct_path, student_path)

            print(f"Processing completed - Marks: {marks}, Rank: {rank}")

            # Handle different types of rank return values
            if isinstance(rank, np.ndarray):
                rank_value = int(rank[0]) if len(rank) > 0 else 0
            elif hasattr(rank, '__len__') and not isinstance(rank, str):
                rank_value = int(rank[0]) if len(rank) > 0 else 0
            else:
                rank_value = int(rank)

            return jsonify({
                'success': True,
                'marks': int(marks),
                'rank': rank_value
            })

        except ImportError as import_error:
            print(f"Import error: {str(import_error)}")
            return jsonify({'error': 'OMR processing module not found. Make sure omr_processing.py exists.'}), 500

        except Exception as processing_error:
            print(f"Processing error: {str(processing_error)}")
            print(f"Error type: {type(processing_error)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error processing images: {str(processing_error)}'}), 500

    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

    finally:
        # Clean up uploaded files
        try:
            if correct_path and os.path.exists(correct_path):
                os.remove(correct_path)
                print(f"Cleaned up: {correct_path}")
        except Exception as cleanup_error:
            print(f"Error cleaning up correct file: {cleanup_error}")

        try:
            if student_path and os.path.exists(student_path):
                os.remove(student_path)
                print(f"Cleaned up: {student_path}")
        except Exception as cleanup_error:
            print(f"Error cleaning up student file: {cleanup_error}")


@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'OMR API is running'})


if __name__ == '__main__':
    print("Starting OMR Evaluation System...")
    print("Make sure you have 'omr_processing.py' with your OMR code in the same directory")
    print("Access the web interface at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)