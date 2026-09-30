#!/bin/bash
# render-build.sh
# Downloads model weights during Render's build step, since they're too
# large to commit to git (*.pth is excluded in .gitignore on purpose).
#
# VGG_URL and DECODER_URL must be DIRECT download links, not a preview page.
# On Hugging Face, use /resolve/main/... - NOT /blob/main/..., which serves
# an HTML preview page instead of the real file. The repo must also be
# Public; a Private repo returns a login page to an unauthenticated wget.
# Correct format:
#   https://huggingface.co/<username>/<repo>/resolve/main/vgg_normalised.pth

set -e  # stop immediately if any step fails, instead of deploying silently broken

VGG_URL="https://huggingface.co/roshanjpr80/restyle-weights/resolve/main/vgg_normalised.pth"
DECODER_URL="https://huggingface.co/roshanjpr80/restyle-weights/resolve/main/decoder_final.pth"

# Fail early with a clear message if either value is not a real web link.
for url in "$VGG_URL" "$DECODER_URL"; do
    if [[ "$url" != https://* ]]; then
        echo "ERROR: '$url' is not an https:// download link."
        echo "Upload the file to Hugging Face (or similar) and paste its public URL in render-build.sh."
        exit 1
    fi
    if [[ "$url" == *"/blob/"* ]]; then
        echo "ERROR: '$url' uses /blob/ (a preview page), not the raw file."
        echo "Change /blob/ to /resolve/ in the URL."
        exit 1
    fi
done

# Downloads a file and verifies it's actually binary model data, not an
# HTML error/login page - the exact failure mode that caused a confusing
# pickle error deep inside PyTorch instead of a clear message here.
download_and_verify() {
    local url="$1"
    local out_file="$2"
    local min_bytes="$3"

    echo "Downloading $out_file..."
    wget -q -O "$out_file" "$url"

    local size
    size=$(stat -c%s "$out_file" 2>/dev/null || stat -f%z "$out_file")

    if [[ "$size" -lt "$min_bytes" ]]; then
        echo "ERROR: $out_file is only $size bytes (expected at least $min_bytes)."
        echo "This almost always means the URL returned a webpage instead of the real file."
        echo "First 200 characters of what was actually downloaded:"
        head -c 200 "$out_file"
        echo ""
        echo "Check: are you using /resolve/main/ (not /blob/main/)? Is the Hugging Face repo Public?"
        exit 1
    fi
    echo "OK: $out_file is $size bytes."
}

download_and_verify "$VGG_URL" "vgg_normalised.pth" 50000000
download_and_verify "$DECODER_URL" "decoder_final.pth" 1000000

echo "Model weights downloaded:"
ls -lh vgg_normalised.pth decoder_final.pth