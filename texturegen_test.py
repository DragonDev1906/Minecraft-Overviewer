import os

from overviewer_core import texturegen
from overviewer_core.textures import Textures
import logging
from logging.config import dictConfig
import sys

class ColorLevelnameFilter:
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"
    COLOR = {
        "DEBUG": CYAN
    }
    HIGHLIGHT = {
        'CRITICAL': RED,
        'ERROR': RED,
        'WARNING': YELLOW,
    }

    def filter(self, record):
        if record.levelname in self.COLOR:
            record.levelname = self.COLOR_SEQ % (30+self.COLOR[record.levelname]) + record.levelname + self.RESET_SEQ
            record.start = ""
            record.end = ""
        elif record.levelname in self.HIGHLIGHT:
            record.start = self.COLOR_SEQ % (40+self.HIGHLIGHT[record.levelname])
            record.end = self.RESET_SEQ
        return True

dictConfig({
    "version": 1,
    "filters": {
        "color_levelname": {
            "()": ColorLevelnameFilter
        }
    },
    "formatters": {
        'standard': {
            'format': '%(start)s%(filename)s:%(lineno)d %(process)d %(asctime)s %(levelname)-7s %(message)s%(end)s'
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            'filters': ['color_levelname'],
            "formatter": "standard",
            'stream': 'ext://sys.stdout',  # Default is stderr
            "level": "DEBUG",
        }
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
        "overviewer_core": {
            "handlers": ["console"],
            "level": "DEBUG",
        },
        "PIL": {
            "handlers": ["console"],
            "level": "INFO"
        },
    }
})

logger = logging.getLogger(__name__)

"""
A way to show multiple images at the same time (up to 100 textures)
"""
from PIL import Image

combine_image_width = 25
combine_image_height = 30
combine_image = Image.new('RGBA', (24 * combine_image_width, 24 * combine_image_height))
_combine_image_current_pos = 0


def add_img(_img):
    global _combine_image_current_pos
    y, x = divmod(_combine_image_current_pos, combine_image_width)
    if _img is not None:
        combine_image.paste(_img, (x * 24, y * 24))
    _combine_image_current_pos += 1


"""
Main Code
"""
base_dir = "/mnt/c/Users/Jens/Documents/projects/Minecraft-Overviewer-working-dir/mc_jar_file"
textures_instance = Textures(base_dir)
block_names = [
    fn.split('.', 1)[0]
    for _, _, files in os.walk(os.path.join(base_dir, texturegen.BlockRenderer.BLOCKSTATES_DIR))
    for fn in files
    if fn.split('.', 1)[1] == "json"
]
block_renderer = texturegen.BlockRenderer(textures_instance)  # , block_list=block_names)


"""
print(block_renderer.guess_size())
prev = ""
for block_name, nbt_index, nbt_condition, variants in block_renderer.iter_blocks(block_names):
    if block_name != prev:
        print('"%s",' % block_name)
        prev = block_name
    for img, weight in variants:
        add_img(img)
        break
"""
# print("Size if all textures could be used:", texturegen.BlockRenderer(
#     textures_instance, block_list=block_names).get_max_size()
#       )
# print("Size with current block:", block_renderer.get_max_size())
for (blockid, data), img in block_renderer.iter_for_generate():
    add_img(img)

"""
Final output of combined texture
"""
combine_image.save("/mnt/c/Users/Jens/Documents/projects/Minecraft-Overviewer-working-dir/tilemap.png")