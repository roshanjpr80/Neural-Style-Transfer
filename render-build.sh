#!/bin/bash
# render-build.sh
# Downloads model weights during Render's build step, since they're too
# large to commit to git (*.pth is excluded in .gitignore on purpose).
#
# Fill in VGG_URL and DECODER_URL below with direct-download links from
# wherever you hosted the files (Hugging Face Hub, Google Drive, etc).
# A Hugging Face file URL looks like:
#   https://huggingface.co/<username>/<repo>/resolve/main/vgg_normalised.pth

set -e  # stop immediately if any step fails, instead of deploying silently broken

VGG_URL="https://huggingface.co/roshanjpr80/restyle-weights/blob/main/vgg_normalised.pth"
DECODER_URL="https://huggingface.co/roshanjpr80/restyle-weights/blob/main/decoder_final.pth"

# Fail early with a clear message if the placeholders were never replaced.
if [[ "$VGG_URL" == REPLACE_WITH_* ]] || [[ "$DECODER_URL" == REPLACE_WITH_* ]]; then
    echo "ERROR: Open render-build.sh and replace VGG_URL and DECODER_URL with your real download links."
    exit 1
fi

echo "Downloading vgg_normalised.pth..."
wget -q -O vgg_normalised.pth "$VGG_URL"

echo "Downloading decoder checkpoint..."
wget -q -O decoder_final.pth "$DECODER_URL"

echo "Model weights downloaded:"
ls -lh vgg_normalised.pth decoder_final.pth
