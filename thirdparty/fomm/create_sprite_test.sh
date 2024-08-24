echo FRONT
python create_spritesheet.py --config /Users/tashvit/Documents/GitHub/mmpixagen/thirdparty/fomm/config/tinyhero-256.yaml \
--checkpoint 'log/tinyhero-256 18_08_24_18.36.14/00000009-checkpoint.pth.tar' \
--source_image data/spritesheet_input/front.png \
--driver data/spritesheet-driver/front \
--result sheet-front.png
echo BACK
python create_spritesheet.py --config /Users/tashvit/Documents/GitHub/mmpixagen/thirdparty/fomm/config/tinyhero-256.yaml \
--checkpoint 'log/tinyhero-256 18_08_24_18.36.14/00000009-checkpoint.pth.tar' \
--source_image data/spritesheet_input/back.png \
--driver data/spritesheet-driver/back \
--result sheet-back.png
echo LEFT
python create_spritesheet.py --config /Users/tashvit/Documents/GitHub/mmpixagen/thirdparty/fomm/config/tinyhero-256.yaml \
--checkpoint 'log/tinyhero-256 18_08_24_18.36.14/00000009-checkpoint.pth.tar' \
--source_image data/spritesheet_input/left.png \
--driver data/spritesheet-driver/left \
--result sheet-left.png
echo RIGHT
python create_spritesheet.py --config /Users/tashvit/Documents/GitHub/mmpixagen/thirdparty/fomm/config/tinyhero-256.yaml \
--checkpoint 'log/tinyhero-256 18_08_24_18.36.14/00000009-checkpoint.pth.tar' \
--source_image data/spritesheet_input/right.png \
--driver data/spritesheet-driver/right \
--result sheet-right.png