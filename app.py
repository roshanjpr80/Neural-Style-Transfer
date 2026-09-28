import os
import time
import uuid

import torch
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap  # provided by the Bootstrap-Flask package
from werkzeug.utils import secure_filename
from wtforms import FileField, SubmitField, FloatField, HiddenField
from wtforms.validators import NumberRange
from PIL import Image
from torchvision import transforms

# Flat file layout (models.py, utils.py next to this file) - not a nested
# 'utils' package.
from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


# Configuration

# Never commit a real secret key to source control. Set SECRET_KEY as an
# environment variable in any real deployment; this fallback is only for
# quick local development and is not safe to share or deploy with.
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-only-insecure-key-change-me')

VGG_PATH = os.environ.get('VGG_PATH', 'vgg_normalised.pth')
# NOTE: decoder_final.pth (as of the version checked into this project) was
# saved at iteration 80 of training - far too early for good stylization
# quality (see train.py notes: meaningful quality typically starts around
# 5,000-20,000 iterations). This is fine for confirming the pipeline works,
# but swap in a later checkpoint (e.g. decoder_iter_20000.pth) once a full
# training run has been done, before using output from this app in your report.
DECODER_PATH = os.environ.get('DECODER_PATH', 'D:/Mojar Project/Neural style transfer/experiment/final_run/decoder_final.pth')

# Longest side any uploaded image is resized to before inference. Keeps a
# single oversized upload from making a request extremely slow or
# memory-heavy, especially on CPU.
MAX_IMAGE_DIM = 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES
Bootstrap(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0, validators=[NumberRange(min=0.0, max=1.0)])
    submit = SubmitField('Transfer Style')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Using device: {device}")


def load_models():
    if not os.path.exists(VGG_PATH):
        raise FileNotFoundError(
            f"VGG weights not found at '{VGG_PATH}'. Set the VGG_PATH environment "
            f"variable or place vgg_normalised.pth next to app.py."
        )
    if not os.path.exists(DECODER_PATH):
        raise FileNotFoundError(
            f"Decoder checkpoint not found at '{DECODER_PATH}'. Set the DECODER_PATH "
            f"environment variable, or train one first with train.py."
        )

    encoder = VGGEncoder(VGG_PATH).to(device)
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load(DECODER_PATH, map_location=device, weights_only=False))

    encoder.eval()
    decoder.eval()
    print(f"[startup] Loaded VGG weights from {VGG_PATH}")
    print(f"[startup] Loaded decoder checkpoint from {DECODER_PATH}")
    return encoder, decoder


encoder, decoder = load_models()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def unique_filename(original_filename):
    """Prefix with a short random id so two users uploading 'photo.jpg' at
    the same time can't silently overwrite each other's files."""
    safe_name = secure_filename(original_filename)
    return f"{uuid.uuid4().hex[:8]}_{safe_name}"


def resize_to_max_dim(image, max_dim):
    """Resize so the longer side is at most max_dim, preserving aspect ratio.
    Only downscales - never enlarges a smaller image."""
    w, h = image.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return image


def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_image = resize_to_max_dim(content_image, MAX_IMAGE_DIM)
    style_image = resize_to_max_dim(style_image, MAX_IMAGE_DIM)

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_image).unsqueeze(0).to(device)
    style_tensor = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if request.method == 'POST' and form.validate_on_submit():
        if form.content.data and form.content.data.filename:
            if allowed_file(form.content.data.filename):
                content_filename = unique_filename(form.content.data.filename)
                form.content.data.save(os.path.join(app.config['UPLOAD_FOLDER'], content_filename))
                form.content_path.data = content_filename
            else:
                error = 'Content image must be a .png, .jpg, or .jpeg file.'
        else:
            content_filename = form.content_path.data

        if form.style.data and form.style.data.filename:
            if allowed_file(form.style.data.filename):
                style_filename = unique_filename(form.style.data.filename)
                form.style.data.save(os.path.join(app.config['UPLOAD_FOLDER'], style_filename))
                form.style_path.data = style_filename
            else:
                error = 'Style image must be a .png, .jpg, or .jpeg file.'
        else:
            style_filename = form.style_path.data

        if not error:
            if not content_filename:
                error = 'Please upload a content image.'
            elif not style_filename:
                error = 'Please upload a style image.'

        if not error:
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

            try:
                content_img = Image.open(content_path).convert('RGB')
                style_img = Image.open(style_path).convert('RGB')

                alpha = min(1.0, max(0.0, float(form.alpha.data)))

                start_time = time.time()
                stylized_image = style_transfer(content_img, style_img, encoder, decoder, alpha, device)
                elapsed = time.time() - start_time

                result_filename = 'stylized_' + content_filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                save_image(stylized_image, result_path)

                result_image = result_filename
                app.logger.info(f"Stylized {content_filename} + {style_filename} in {elapsed:.2f}s")
            except Exception as e:
                app.logger.exception("Style transfer failed")
                error = "Something went wrong generating the image. Please try different files."
    elif request.method == 'POST':
        # form.validate_on_submit() was False on an actual POST (e.g. a bad
        # or missing CSRF token) - a real error, unlike a plain GET.
        error = 'Your session expired or the form was invalid. Please try again.'

    return render_template(
        'index.html', form=form, result_image=result_image, content_image=content_filename,
        style_image=style_filename, error=error,
    )


@app.errorhandler(413)
def file_too_large(e):
    return render_template(
        'index.html', form=UploadForm(), result_image=None, content_image=None,
        style_image=None, error=f"File too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
    ), 413


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory('examples', filename)


if __name__ == '__main__':
    from werkzeug.serving import run_simple

    # Debug mode enables Werkzeug's interactive in-browser debugger, which
    # can execute arbitrary code if an error page is ever reached - fine for
    # solo local development, not safe to leave on for anything reachable by
    # anyone else. Off by default; opt in explicitly.
    debug_mode = os.environ.get('FLASK_DEBUG', '0') == '1'
    run_simple('localhost', 5000, app, use_reloader=debug_mode, use_debugger=debug_mode)