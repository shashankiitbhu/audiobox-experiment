# Run this on cloud to get audio files

echo "Downloading ESC-50 dataset..."
wget https://github.com/karolpiczak/ESC-50/archive/master.zip
unzip -q master.zip
mkdir -p audio
cp ESC-50-master/audio/*.wav audio/
rm -rf ESC-50-master master.zip

echo "✅ Audio files ready in audio/"
ls audio/ | wc -l
EOF

chmod +x scripts/setup_data.sh