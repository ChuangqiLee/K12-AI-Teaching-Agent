#!/usr/bin/env bash
# Clone and prepare the upstream projects the agent builds on.
#   - Wav2Lip                     (Prajwal et al., 2020)      -> third_party/Wav2Lip
#   - Real-Time-Voice-Cloning     (English, CorentinJ)        -> third_party/Real-Time-Voice-Cloning
#   - MockingBird (optional)      (Mandarin fork of RTVC)     -> third_party/MockingBird   [--zh]
#
# Usage: bash scripts/setup_third_party.sh [--zh]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TP="$ROOT/third_party"
mkdir -p "$TP"
cd "$TP"

# ------------------------------------------------------------------ Wav2Lip
if [ ! -d Wav2Lip ]; then
  git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git
fi
# librosa >= 0.10 only accepts keyword arguments in filters.mel
sed -i.bak 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' Wav2Lip/audio.py
mkdir -p Wav2Lip/checkpoints Wav2Lip/face_detection/detection/sfd
if [ ! -f Wav2Lip/face_detection/detection/sfd/s3fd.pth ]; then
  curl -L -o Wav2Lip/face_detection/detection/sfd/s3fd.pth \
    https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth || \
    echo "!! download s3fd.pth manually into Wav2Lip/face_detection/detection/sfd/"
fi
echo ">> Wav2Lip: put wav2lip_gan.pth (and lipsync_expert.pth for training) into third_party/Wav2Lip/checkpoints/"
echo "   (links: https://github.com/Rudrabha/Wav2Lip#getting-the-weights)"

# ------------------------------------------------------------------ Real-Time-Voice-Cloning
if [ ! -d Real-Time-Voice-Cloning ]; then
  git clone --depth 1 https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
fi
pip install inflect unidecode "webrtcvad-wheels" tqdm >/dev/null
( cd Real-Time-Voice-Cloning && python -c "
from pathlib import Path
from utils.default_models import ensure_default_models
ensure_default_models(Path('saved_models'))
print('>> RTVC pretrained encoder / synthesizer / vocoder in saved_models/default/')
" ) || echo "!! could not auto-download RTVC models; see the RTVC wiki"

# ------------------------------------------------------------------ MockingBird (Mandarin)
if [ "${1:-}" = "--zh" ]; then
  if [ ! -d MockingBird ]; then
    git clone --depth 1 https://github.com/babysor/MockingBird.git
  fi
  pip install pypinyin >/dev/null
  echo ">> MockingBird: download a Mandarin synthesizer + HiFi-GAN vocoder (see its README) and set"
  echo "   speech.repo_dir / *_ckpt in configs/zh_mockingbird.yaml"
fi
echo "done."
