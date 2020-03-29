import os

from overviewer_core import texturegen
from overviewer_core.textures import Textures

"""
A way to show multiple images at the same time (up to 100 textures)
"""
from PIL import Image
combine_image_width = 25
combine_image_height = 30
combine_image = Image.new('RGBA', (25 * combine_image_width, 25 * combine_image_height))
_combine_image_current_pos = 0


def add_img(_img):
    global _combine_image_current_pos
    y, x = divmod(_combine_image_current_pos, combine_image_width)
    if _img is not None:
        combine_image.paste(_img, (x * 25, y * 25))
    _combine_image_current_pos += 1


"""
Main Code
"""
def gen_tileset(outputdir, texturepath=None):


    # base_dir = "C:/Users/Lonja/AppData/Roaming/.minecraft/versions/20w12a/20w12a.jar"
    textures_instance = Textures(texturepath)
    # block_names = [
    #     fn.split('.', 1)[0]
    #     for _, _, files in os.walk(os.path.join(base_dir, texturegen.BlockRenderer.BLOCKSTATES_DIR))
    #     for fn in files
    #     if fn.split('.', 1)[1] == "json"
    #
    # ]
    # print(block_names)
    block_renderer = texturegen.BlockRenderer(textures_instance)  # , block_list=block_names)

    # print("Size if all textures could be used:", texturegen.BlockRenderer(
    #     textures_instance, block_list=block_names).get_max_size()
    #       )
    # print(list(block_renderer.iter_all_blocks()))
    print("Size with current block:", block_renderer.get_max_size())
    for (blockid, data), img in block_renderer.iter_for_generate():
        # print("hi")
        add_img(img)

    """
    Final output of combined texture
    """
    combine_image.save(
        os.path.join(outputdir, "tileset.png"))