from collections import defaultdict
from functools import lru_cache

from PIL import Image
import logging

from asset_loader import AssetLoaderMixin

START_BLOCK_ID = 20000
# BLOCK_LIST = [
#     "dirt",
#     "cyan_stained_glass",
#     "gray_concrete_powder",
#     "red_terracotta",
#     "quartz_block",
#     "beehive",
#     "birch_leaves",
#     "gray_concrete",
#     "brown_glazed_terracotta",
#     "magma_block",
#     "hay_block",
#     "oak_log",
#     "end_stone_bricks",
#     "iron_ore",
#     "emerald_block",
#     "light_blue_concrete_powder",
#     "black_concrete_powder",
#     "jigsaw",
#     "dispenser",
#     "green_terracotta",
#     "dead_brain_coral_block",
#     "red_concrete",
#     "mossy_cobblestone",
#     "barrel",
#     "repeating_command_block",
#     "stripped_jungle_log",
#     "magenta_concrete",
#     "gray_wool",
#     "diamond_ore",
#     "orange_stained_glass",
#     "white_wool",
#     "blue_ice",
#     "terracotta",
#     "redstone_lamp",
#     "dark_oak_planks",
#     "wither_skeleton_wall_skull",
#     "polished_andesite",
#     "ice",
#     "purple_glazed_terracotta",
#     "jungle_log",
#     "prismarine",
#     "light_gray_concrete",
#     "red_sandstone",
#     "orange_concrete_powder",
#     "coarse_dirt",
#     "gold_block",
#     "light_blue_terracotta",
#     "wet_sponge",
#     "white_glazed_terracotta",
#     "lime_terracotta",
#     "white_concrete",
#     "green_wool",
#     "birch_planks",
#     "purpur_block",
#     "cracked_stone_bricks",
#     "lapis_block",
#     "iron_block",
#     "magenta_concrete_powder",
#     "black_stained_glass",
#     "chiseled_red_sandstone",
#     "spruce_planks",
#     "coal_ore",
#     "red_nether_bricks",
#     "crafting_table",
#     "stone_bricks",
#     "black_concrete",
#     "stripped_jungle_wood",
#     "smoker",
#     "pink_wool",
#     "diamond_block",
#     "black_glazed_terracotta",
#     "chiseled_sandstone",
#     "dark_oak_wood",
#     "mycelium",
#     "gray_glazed_terracotta",
#     "purpur_pillar",
#     "lime_wool",
#     "cyan_concrete_powder",
#     "infested_cracked_stone_bricks",
#     "stripped_spruce_log",
#     "yellow_wool",
#     "dead_bubble_coral_block",
#     "oak_planks",
#     "white_concrete_powder",
#     "dark_oak_leaves",
#     "emerald_ore",
#     "jack_o_lantern",
#     "pink_terracotta",
#     "podzol",
#     "nether_quartz_ore",
#     "andesite",
#     "end_stone",
#     "chiseled_stone_bricks",
#     "sponge",
#     "stripped_acacia_wood",
#     "furnace",
#     "brown_concrete",
#     "cut_sandstone",
#     "red_sand",
#     "light_blue_wool",
#     "cyan_concrete",
#     "infested_mossy_stone_bricks",
#     "light_blue_stained_glass",
#     "cyan_glazed_terracotta",
#     "acacia_planks",
#     "blast_furnace",
#     "netherrack",
#     "nether_bricks",
#     "piston",
#     "infested_stone_bricks",
#     "fire_coral_block",
#     "skeleton_wall_skull",
#     "snow_block",
#     "stripped_oak_wood",
#     "lime_concrete_powder",
#     "light_gray_wool",
#     "gray_stained_glass",
#     "lime_concrete",
#     "quartz_pillar",
#     "dead_tube_coral_block",
#     "acacia_leaves",
#     "cartography_table",
#     "red_glazed_terracotta",
#     "dropper",
#     "birch_log",
#     "red_stained_glass",
#     "diorite",
#     "smithing_table",
#     "glowstone",
#     "melon",
#     "wither_skeleton_skull",
#     "yellow_terracotta",
#     "light_gray_stained_glass",
#     "prismarine_bricks",
#     "cyan_wool",
#     "jukebox",
#     "white_terracotta",
#     "mossy_stone_bricks",
#     "green_concrete",
#     "orange_concrete",
#     "pink_concrete",
#     "blue_concrete",
#     "brown_concrete_powder",
#     "white_stained_glass",
#     "sand",
#     "stripped_dark_oak_wood",
#     "yellow_stained_glass",
#     "cobblestone",
#     "chain_command_block",
#     "lapis_ore",
#     "acacia_log",
#     "spruce_wood",
#     "spruce_leaves",
#     "smooth_stone",
#     "oak_wood",
#     "oak_leaves",
#     "purple_concrete_powder",
#     "gray_terracotta",
#     "yellow_concrete_powder",
#     "note_block",
#     "magenta_stained_glass",
#     "smooth_quartz",
#     "light_gray_glazed_terracotta",
#     "magenta_glazed_terracotta",
#     "infested_cobblestone",
#     "clay",
#     "horn_coral_block",
#     "glass",
#     "stripped_spruce_wood",
#     "lime_glazed_terracotta",
#     "purple_wool",
#     "smooth_red_sandstone",
#     "light_blue_concrete",
#     "obsidian",
#     "brown_wool",
#     "blue_stained_glass",
#     "light_blue_glazed_terracotta",
#     "red_wool",
#     "jungle_planks",
#     "purple_terracotta",
#     "gravel",
#     "blue_concrete_powder",
#     "light_gray_terracotta",
#     "cut_red_sandstone",
#     "dark_prismarine",
#     "black_terracotta",
#     "loom",
#     "yellow_concrete",
#     "light_gray_concrete_powder",
#     "soul_sand",
#     "pink_concrete_powder",
#     "command_block",
#     "sea_lantern",
#     "spruce_log",
#     "skeleton_skull",
#     "bubble_coral_block",
#     "lime_stained_glass",
#     "purple_stained_glass",
#     "jungle_leaves",
#     "sandstone",
#     "fletching_table",
#     "green_stained_glass",
#     "polished_granite",
#     "dead_horn_coral_block",
#     "stripped_birch_wood",
#     "bee_nest",
#     "brown_stained_glass",
#     "yellow_glazed_terracotta",
#     "cyan_terracotta",
#     "green_concrete_powder",
#     "pink_glazed_terracotta",
#     "black_wool",
#     "purple_concrete",
#     "bone_block",
#     "blue_terracotta",
#     "stripped_dark_oak_log",
#     "green_glazed_terracotta",
#     "redstone_ore",
#     "blue_wool",
#     "dark_oak_log",
#     "orange_terracotta",
#     "packed_ice",
#     "chiseled_quartz_block",
#     "orange_glazed_terracotta",
#     "bricks",
#     "honeycomb_block",
#     "structure_block",
#     "sticky_piston",
#     "acacia_wood",
#     "bookshelf",
#     "pumpkin",
#     "stripped_oak_log",
#     "stripped_acacia_log",
#     "blue_glazed_terracotta",
#     "frosted_ice",
#     "redstone_block",
#     "orange_wool",
#     "brain_coral_block",
#     "gold_ore",
#     "nether_wart_block",
#     "jungle_wood",
#     "coal_block",
#     "magenta_terracotta",
#     "smooth_sandstone",
#     "tube_coral_block",
#     "stripped_birch_log",
#     "polished_diorite",
#     "magenta_wool",
#     "spawner",
#     "dead_fire_coral_block",
#     "pink_stained_glass",
#     "carved_pumpkin",
#     "red_concrete_powder",
#     "granite",
#     "birch_wood",
#     "brown_terracotta",
#     "infested_chiseled_stone_bricks",
#     "tnt",
# ]

################################################################
# Constants and helper methods for the BlockRenderer
# Placed here in order to not need self.
################################################################
N = "north"
E = "east"
S = "south"
W = "west"
U = "up"
D = "down"


def rot(image, degree):
    if image is None:
        return None
    return image.rotate(degree)


def flip(image, arg1):
    if image is not None:
        return image.transpose(arg1)
    else:
        return None
logger = logging.getLogger(__name__)

################################################################
# Main Code for Block Rendering
################################################################


class BlockRenderer(AssetLoaderMixin, object):
    BLOCKSTATES_DIR = "assets/minecraft/blockstates"
    MODELS_DIR = "assets/minecraft/models"
    TEXTURES_DIR = "assets/minecraft/textures"
    data_map = defaultdict(list)

    def __init__(self, textures, *, block_list:list=None, start_block_id=1):
        self.textures = textures
        self.start_block_id = start_block_id
        if block_list is None:
            self.block_list = self.walk_assets(self.BLOCKSTATES_DIR,
                                                 r".json")
        else: self.block_list = block_list

    ################################################################
    # Loading files
    ################################################################

    ################################################################
    # Model file parsing
    ################################################################
    @staticmethod
    def combine_textures_fields(textures_field: dict, parent_textures_field: dict) -> dict:
        return {
            **{
                key: textures_field.get(value[1:], value) if value[0] == '#' else value
                for key, value in parent_textures_field.items()
            },
            **textures_field
        }

    def load_and_combine_model(self, name):
        data = self.load_model(name)
        if "parent" in data:
            # Load the JSON from the parent
            parent_data = self.load_and_combine_model(data["parent"])
            elements_field = data.get("elements", parent_data.get("elements"))
            textures_field = self.combine_textures_fields(data.get("textures", {}), parent_data.get("textures", {}))
        else:
            elements_field = data.get("elements")
            textures_field = data.get("textures", {})

        return {
            "textures": textures_field,
            "elements": elements_field,
        }

    def get_max_nbt_count(self, name: str) -> int:
        data = self.load_blockstates(name)
        return len(data.get("variants", []))

    ################################################################
    # Render methods
    ################################################################
    @lru_cache
    def load_img(self, texture_name):
        with self.load_file(self.TEXTURES_DIR, texture_name, ".png") as f:
            buffer =  Image.open(f)
            # buffer = BytesIO(fileobj.read())
            h, w =buffer.size #get images size
            if h != w:# check image is square if not (for example due to animated texture) crop shorter side
                buffer = buffer.crop((0,0,min(h,w),min(h,w)))
            texture = buffer.convert("RGBA")
            return texture
    def render_single_cube(self, part, textures, rotation_x_axis, rotation_y_axis):
        """
        Limitations:
        - Only full blocks
        - Rotation is not supported
        - UV coordinates must be [0, 0, 16, 16] (can be implemented bt isn't yet) \
        - tintcolor
        """
        s = defaultdict(lambda: None)
        for face_name, definition in part["faces"].items():
            # uv, texture, cullface, rotation, tintindex
            # Get the Texture (in case no variable is use
            texture_name = textures[definition["texture"][1:]]


            texture = self.load_img(texture_name)

            if "rotation" in definition:
                texture = texture.rotate(-definition["rotation"])
            s[face_name] = texture

        # Test for applying the correct top side for observers
        # s[U] = flip(s[U], Image.FLIP_TOP_BOTTOM)

        # Fix rotation of the bottom texture (see Assumption)
        s[D] = rot(s[D], 180)

        # TODO: Optimize by only rotating textures if needed (store rotation and apply it later)

        # Apply Rotation in X axis by swapping and optionally rotating the textures
        if rotation_x_axis == 90:
            # Identified using the Observer
            s[N], s[S], s[U], s[D] = rot(s[U], 180), rot(s[D], 180), rot(s[S], 180), s[N]
        elif rotation_x_axis == 180:
            # Identifies using the Barrel Block
            s[N], s[S], s[U], s[D] = rot(s[S], 180), rot(s[N], 180), rot(s[D], 180), rot(s[U], 180)
        elif rotation_x_axis == 270:
            # Identified using the Observer
            s[N], s[S], s[U], s[D] = s[D], s[U], s[N], rot(s[S], 180)
        s[E] = rot(s[E], -rotation_x_axis)  # Identified using the Observer
        s[W] = rot(s[W], rotation_x_axis)  # Identified using the Observer

        # Apply Rotation in Y axis by swapping and optionally rotating the textures
        if rotation_y_axis == 90:
            # Identified using the Observer
            s[N], s[E], s[S], s[W] = s[W], s[N], s[E], s[S]
        elif rotation_y_axis == 180:
            # Identified using the Observer
            s[N], s[E], s[S], s[W] = s[S], s[W], s[N], s[E]
        elif rotation_y_axis == 270:
            # Identified using the Observer
            s[N], s[E], s[S], s[W] = s[E], s[S], s[W], s[N]
        s[U] = rot(s[U], -rotation_y_axis)  # Identified using the Observer
        s[D] = rot(s[D], rotation_y_axis)  # Identified using the Observe

        # Rotation and Flipping identified by passing one side to the function
        # with an easily identifiable texture
        return self.textures.build_full_block(
            s.get("up"),
            s.get("north"),
            flip(s.get("east"), Image.FLIP_LEFT_RIGHT),
            s.get("west"),
            flip(s.get("south"), Image.FLIP_LEFT_RIGHT),
            flip(s.get("down"), Image.FLIP_LEFT_RIGHT)
        )

    def render_model(self, data: dict, rotation_x_axis, rotation_y_axis, uvlock):
        """
        This method is currently only using existing draw methods. Because of that only full size blocks can be created.

        Assumption:
        - The bottom texture is pointing in the reverse direction of the top texture
        Reason: models/blocks/template_glaced_terracotta.json does not rotate the top or bottom faces
        ==> This Indicates the default rotation

        Limitations:
        - Only full blocks
        - Rotation is not supported
        - UV coordinates must be [0, 0, 16, 16] (can be implemented bt isn't yet)
        - tintcolor
        """
        # Check if there are errors and the block can't be rendered correctly because of limitations listed above
        if data["elements"] is None or len(data["elements"]) == 0:
            raise NotImplementedError("Only blocks with 'elements' are supported.")

        for part in data["elements"]:
            if part["from"] != [0, 0, 0] or part["to"] != [16, 16, 16]:
                raise NotImplementedError("Partial Blocks are not supported")
            if "rotation" in part:
                raise NotImplementedError("Rotated Elements are not supported")

            for face_name, definition in part["faces"].items():
                if "uv" in definition and definition["uv"] != [0, 0, 16, 16]:
                    raise NotImplementedError("Only elements with UV [0, 0, 16, 16] are supported")

        if len(data["elements"]) != 1:
            raise NotImplementedError("Blocks with multiple parts are not supported because the current drawing "
                                      "can't identify which parts should be drawn on top.")

        # Render the parts in the order they are in the file.
        # Reason: Required for correct rendering of e.g. the grass block
        for part in data["elements"]:
            img = self.render_single_cube(part, data["textures"], rotation_x_axis, rotation_y_axis)
            return img

        raise RuntimeError("This code should not be reachable. If support for multiple parts is added, "
                           "the combined texture is returned here. Until then the checks above should "
                           "ensure this code is never reached.")

    def render_blockstate_entry(self, data: dict) -> Image:
        # model, x, y, uvlock, weight
        return self.render_model(
            data=self.load_and_combine_model(data["model"]),
            rotation_x_axis=data.get("x", 0),  # Increments of 90°
            rotation_y_axis=data.get("y", 0),  # Increments of 90°
            uvlock=data.get("uvlock", False)
        ), data.get("weight", 1)

    def render_blockstates(self, data: dict) -> (str, []):
        if "variants" in data:
            for nbt_condition, variant in data["variants"].items():
                yield (nbt_condition, [
                    self.render_blockstate_entry(v)
                    for v in (variant if type(variant) == list else (variant,))
                ])
        else:
            raise NotImplementedError("Multipart is not supported. Only blocks using 'variants' are.")

    ################################################################
    # NBT to Data conversion
    ################################################################
    @staticmethod
    def store_nbt_as_int(name, nbt_condition, blockid, data_value):
        compare_set = {x.split('=')[0] for x in nbt_condition.split(',') if x != ""}
        BlockRenderer.data_map["minecraft:%s" % name].append((compare_set, (blockid, data_value)))

    @staticmethod
    def get_nbt_as_int(key: str, properties: dict):
        entry = BlockRenderer.data_map.get(key)
        if entry is None:
            return None, None

        if properties is None:
            properties = {}

        for stored_set, data_value in entry:
            for x in properties.items():
                if "%s=%s" % x not in stored_set:
                    break
            else:
                return data_value
        # If nothing matches return something (better than drawing nothing)
        # This should only happen if the BlockRenderer can't render a variant
        if len(entry) > 0:
            return entry[0][1]
        else:
            # No valid variant found (probably won't ever happen because no entry would be created)
            return None, None

    ################################################################
    # Genrators and methods for rendering multiple blocks
    ################################################################
    def iter_blocks(self, name_list: list, ignore_unsupported_blocks=True):
        for block_index, block_name in enumerate(name_list):
            try:
                for nbt_index, (nbt_condition, variants) in enumerate(
                        self.render_blockstates(self.load_blockstates(name=block_name))
                ):
                    yield block_index, block_name, nbt_index, nbt_condition, variants
            except NotImplementedError as e:
                if not ignore_unsupported_blocks:
                    raise e

    def iter_all_blocks(self, ignore_unsupported_blocks=True):
        # TODO: This method is currently NOT working because os.walk can't be used inside a jar file
        #todo, can this not be done as a yield iterator?
        logger.debug("iter_all_blocks")
        return self.iter_blocks(sorted(self.walk_assets(self.BLOCKSTATES_DIR, r".json")))

    def get_max_size(self) -> (int, int):
        blockid_count = len(list(self.block_list))
        data_count = max(self.get_max_nbt_count(name) for name in list(self.block_list))
        return blockid_count + self.start_block_id, data_count

    def iter_for_generate(self):
        for block_index, block_name, nbt_index, nbt_condition, variants in self.iter_all_blocks():
            if len(variants) >= 1:
                #print("B:", block_name, block_index + self.start_block_id, nbt_index, nbt_condition)
                BlockRenderer.store_nbt_as_int(block_name, nbt_condition, block_index + self.start_block_id, nbt_index)
                yield (block_index + self.start_block_id, nbt_index), variants[0][0]
