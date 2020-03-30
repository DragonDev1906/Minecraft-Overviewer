import json
import os
import typing
from collections import defaultdict
from functools import lru_cache
from pprint import pprint

import moderngl as mgl
import numpy as np
from math import sin, cos, tan, asin, pi

from overviewer_core.util import get_program_path
from PIL import Image
import logging

logger = logging.getLogger(__name__)
from overviewer_core.asset_loader import AssetLoader

START_BLOCK_ID = 20000
models_built= 0

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


################################################################
# Simple function for loading .obj files
################################################################
def load_obj(ctx, render_program, path):
    """
    This method loads a .obj file and creates a model that can be rendered using moderngl.

    Wavefont (.obj) files reference verticies, uvs and normals for each vertex with different indicies.
    ModernGL does not support different indicies for verticies, uvs and normals.

    In addition to de-referencing the verticies, uvs and normals they are combined into a single VertexBuffer.

    Finally a VertexArray is created, that can be rendered using
    return_value.render(mode=mgl.TRIANGLES)
    """
    # Store all verticies, uvs, normals and faces for later processing
    raw_verticies = []
    raw_uvs = []
    raw_normals = []
    raw_faces = []

    # Read data from the file and store it into the arrays above
    with open(os.path.join(get_program_path(),path), "r") as fp:
        for line in fp.readlines():
            line = line.strip()
            if line[0] == '#':
                continue
            args = tuple(line.split(' ')[1:])
            if line.startswith('v '):
                raw_verticies.append(args)
            elif line.startswith('vt '):
                raw_uvs.append(args)
            elif line.startswith('vn '):
                raw_normals.append(args)
            elif line.startswith('f '):
                raw_faces.append(tuple(x.split('/') for x in args))
            else:
                pass

    # Combine all arrays and do the de-referencing
    data = np.array([
        value
        for face in raw_faces
        for vertex in face
        for value_list in [         # Each vertex consists of a position, normal and a uv
            raw_verticies[int(vertex[0])-1],
            raw_normals[int(vertex[2])-1],
            raw_uvs[int(vertex[1])-1],
        ]
        for value in value_list     # Combine all small arrays into a single large one
    ], dtype="f4")

    # This array is used to ensure faces always have the same id in later code.
    # This should only need to be changed if cube.obj changes
    # The index is the normal-index from cube.obj
    # The value is the index used later for this face
    face_mapping = [
        4,  # Top
        2,  # South
        3,  # West
        5,  # Bottom
        1,  # East
        0   # North
    ]

    print(raw_faces)
    face_indicies = np.array([
        face_mapping[int(vertex[2])-1]
        for face in raw_faces
        for vertex in face
    ], dtype="u4")

    # By reshaping the array all values can be read more easily
    print(data.reshape((data.size // 8, 8)))
    print(face_indicies)

    # Create a buffer containing the data
    cube_vbo = ctx.buffer(data.tobytes())
    cube_vbo_int = ctx.buffer(face_indicies.tobytes())
    # Create a VertexArray bound to the buffer and the render_program
    return ctx.vertex_array(
        render_program,
        [
            (cube_vbo, "3f4 3f4 2f4", "in_vert", "in_normal", "in_texcoord_0"),
            (cube_vbo_int, "u4", "in_faceid")
        ]
    )


################################################################
# Main Code for Block Rendering
################################################################
class BlockRenderer(object):



    # Storage for finding the data value
    data_map = defaultdict(list)

    def __init__(self, textures, *, block_list=None, start_block_id: int=1, resolution: int=24,
                 vertex_shader: str="overviewer_core/rendering/default.vert",
                 fragment_shader: str="overviewer_core/rendering/default.frag",
                 projection_matrix=None, mc_texture_size=16):
        # Not direclty related to rendering

        self.textures = textures
        self.mc_texture_size = mc_texture_size
        self.assetLoader = AssetLoader(textures.find_file_local_path, (self.mc_texture_size, self.mc_texture_size))

        self.start_block_id = start_block_id

        if block_list is None:
            self.block_list = self.assetLoader.get_blocklist()
        else: self.block_list = block_list
        # self._block_textures()
        # Settings for rendering
        self.resolution = resolution

        # Configure rendering
        self.ctx, self.fbo, self.cube_model = self.setup_rendering(vertex_shader, fragment_shader, projection_matrix)

    def setup_rendering(self, vertex_shader, fragment_shader, projection_matrix=None):
        # Read shader source-code

        with open(os.path.join(get_program_path(),vertex_shader)) as fp:
            vertex_shader_src = fp.read()
        with open(os.path.join(get_program_path(),fragment_shader)) as fp:
            fragment_shader_src = fp.read()

        # Setup for rendering
        try:
            # TODO: EGL seems to need some commands first (currently manually executed)
            #  They probably must only be executed once per shell?
            ctx = mgl.create_context(
                standalone=True,
                backend='egl',
                libgl='libGL.so.1',
                libegl='libEGL.so.1',
            )
        except ImportError:
            ctx = mgl.create_context(
                standalone=True,
            )

        # DEPTH_TEST to calculate which face is visible, CULL_FACE to hide the backside of each face
        ctx.enable(mgl.DEPTH_TEST | mgl.CULL_FACE | mgl.BLEND)
        # Create a framebuffer to render into
        fbo = ctx.simple_framebuffer((self.resolution, self.resolution), components=4)
        fbo.use()
        # Compile the shaders
        render_program = ctx.program(vertex_shader=vertex_shader_src, fragment_shader=fragment_shader_src)

        # Load cube.obj to get a model to render (ArrayBuffer)
        cube_vao = load_obj(ctx, render_program, "overviewer_core/rendering/cube.obj")

        # If no projection_matrix is given use the kind-of isometric view used by Overviewer
        if projection_matrix is None:
            projection_matrix = self.calculate_projection_matrix()

        # Load all textures into a single TextureArray
        # All textures have to be of size mc_texture_size*mc_texture_size
        texture_list = self.get_all_textures()
        for tex in texture_list:
            print(tex)

        print(len(texture_list))
        texture_array = ctx.texture_array(
            size=(self.mc_texture_size, self.mc_texture_size, len(texture_list)),
            components=4,
            data=b''.join([
                img.tobytes() for img in texture_list
            ])
        )
        texture_array.filter = mgl.NEAREST, mgl.NEAREST
        texture_array.use()

        # Set the "uniform" values the shaders require
        render_program["Mvp"].write(projection_matrix.astype('f4').tobytes())

        return ctx, fbo, cube_vao

    def calculate_projection_matrix(self):
        # Orthographic view matrix
        top = .8165
        bot = -.8165
        left = -.7073
        right = .7073
        near = -10
        far = 10
        projection_matrix = np.array([
            [2 / (right-left), 0, 0, -(right+left)/(right-left)],
            [0, 2/(top-bot), 0, -(top+bot)/(top-bot)],
            [0, 0, -2/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1],
        ], dtype="f4")

        # Rotation matricies from Wikipedia: https://en.wikipedia.org/wiki/Rotation_matrix
        # Alpha values from Wikipedia: https://en.wikipedia.org/wiki/Isometric_projection#Mathematics
        alpha = asin(tan(pi / 6))
        # print(alpha)
        s = sin(alpha)
        c = cos(alpha)
        rot_x = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ])

        alpha = pi / 4
        s = sin(alpha)
        c = cos(alpha)
        rot_y = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ])

        transform = np.matmul(rot_x, rot_y)
        view_matrix = np.linalg.inv(transform)

        # roty*rotx is the normal isometric view, the scale makes it fit into a square after applying the isometric view
        # The transform matrix must be inverted and combined with the projection_matrix to get the VP matrix
        return np.matmul(view_matrix, projection_matrix)

    ################################################################
    # Finding and indexing textures
    ################################################################



    ################################################################
    # Model file parsing
    ################################################################



    # def get_max_nbt_count(self, name: str) -> int:
    #     data = self.assetLoader.load_blockstates(name)
    #     pprint(data)
    #     return len(data.get("variants", []))

    ################################################################
    # Render methods
    ################################################################
    def render_vertex_array(self, vertex_array: mgl.VertexArray, face_texture_ids: list, face_uvs: list, *,
                            pos=(0, 0, 0), model_rot=(0, 0, 0), scale=(1, 1, 1)):
        # Write uniform values and render the vertex_array
        vertex_array.program["face_texture_ids"].write(np.array(face_texture_ids, dtype="u4").tobytes())
        vertex_array.program["face_uvs"].write(np.array(face_uvs, dtype="f4").tobytes())
        vertex_array.program["model_rot"].write(np.array(model_rot, dtype="f4").tobytes())
        vertex_array.program["pos"].write(np.array(pos, dtype="f4"))
        vertex_array.program["scale"].write(np.array(scale, dtype="f4"))
        vertex_array.render()

    def render_element(self, element, texture_variables: dict, rotation_x_axis, rotation_y_axis, uvlock):
        # Convert the two cube corners into postion, and scale
        pos = tuple((t + f) / 32 - .5 for f, t in zip(element["from"], element["to"]))
        model_rot = (rotation_x_axis / 180 * pi, rotation_y_axis / 180 * pi)     # Not implemented yet
        scale = tuple((t - f) / 16 for f, t in zip(element["from"], element["to"]))
        face_texture_ids = [
            self.get_texture_index(texture_variables[element["faces"][face_name]["texture"][1:]])
            if face_name in element["faces"]
            else 0
            for face_name in ["north", "east", "south", "west", "up", "down"]
        ]
        face_uvs = [
            value / 16
            for face_name in ["north", "east", "south", "west", "up", "down"]
            for value in element["faces"].get(face_name, {}).get("uv", (0, 0, 16, 16))
        ]

        # Render the cube
        self.render_vertex_array(self.cube_model, face_texture_ids, face_uvs, pos=pos, model_rot=model_rot, scale=scale)

    def render_model(self, data: dict, rotation_x_axis, rotation_y_axis, uvlock):
        """
        This method is currently only using existing draw methods. Because of that only full size block can be created.

        Assumption:
        - The bottom texture is pointing in the reverse direction of the top texture
        Reason: models/block/template_glaced_terracotta.json does not rotate the top or bottom faces
        ==> This Indicates the default rotation

        Limitations:
        - Only full block
        - Rotation is not supported
        - UV coordinates must be [0, 0, 16, 16] (can be implemented bt isn't yet)
        - tintcolor
        """
        # Check if there are errors and the block can't be rendered correctly because of limitations listed above
        if data["elements"] is None or len(data["elements"]) == 0:
            raise NotImplementedError("Only block with 'elements' are supported.")

        # Check if the block is currently supported
        for part in data["elements"]:
            if "rotation" in part:
                raise NotImplementedError("Rotated Elements are not supported")

            for face_name, definition in part["faces"].items():
                if "uv" in definition and definition["uv"] != [0, 0, 16, 16]:
                    # raise NotImplementedError("Only elements with UV [0, 0, 16, 16] are supported")
                    pass

        # Clear the renderbuffer to start a new image
        self.ctx.clear()

        # Render the parts in the order they are in the file.
        # Reason: Required for correct rendering of e.g. the grass block
        for part in data["elements"]:
            # Render a single cube
            self.render_element(part, data["textures"], rotation_x_axis, rotation_y_axis, uvlock)

        # Read the data from the framebuffer and return it as a PIL.Image
        img = Image.frombytes("RGBA", (self.resolution, self.resolution), self.fbo.read(components=4))
        return img.transpose(Image.FLIP_TOP_BOTTOM)

    def render_blockstate_entry(self, data: dict) -> Image:
        # model, x, y, uvlock, weight
        global models_built
        # logger.info("Building models: {0}".format(models_built))
        models_built +=1
        return self.render_model(
            data=self.assetLoader.get_model(data["model"]),
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
            raise NotImplementedError("Multipart is not supported. Only block using 'variants' are.")

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
    # Genrators and methods for rendering multiple block
    ################################################################
    def iter_blocks(self, name_list: list, ignore_unsupported_blocks=True):

        for block_index, block_name in enumerate(name_list):

            print(block_name)
            try:
                for nbt_index, (nbt_condition, variants) in enumerate(
                        self.render_blockstates(self.assetLoader.load_blockstates(name=block_name)[1])
                ):
                    yield block_index, block_name, nbt_index, nbt_condition, variants
            except NotImplementedError as e:
                if not ignore_unsupported_blocks:
                    raise e

    def iter_all_blocks(self, ignore_unsupported_blocks=True):

        logger.debug("Searching for blockstates in " + self.assetLoader.BLOCKSTATES_DIR)

       
        return self.iter_blocks(sorted(self.assetLoader.get_blocklist()))

    # def get_max_size(self) -> (int, int):
    #     blockid_count = len(list(self.block_list))
    #     data_count = max(self.get_max_nbt_count(name) for name in list(self.block_list))
    #     return blockid_count + self.start_block_id, data_count

    def iter_for_generate(self):
        for block_index, block_name, nbt_index, nbt_condition, variants in self.iter_all_blocks():
            if len(variants) >= 1:
                logger.debug("Block found: {0} -> {1}:{2}".format(block_name, block_index, nbt_index))
                BlockRenderer.store_nbt_as_int(block_name, nbt_condition, block_index + self.start_block_id, nbt_index)
                # pprint((block_index + self.start_block_id, nbt_index), variants[0][0])
                yield (block_index + self.start_block_id, nbt_index), variants[0][0]

    # @lru_cache()
    def get_all_textures(self)->list:
        """Wrapper to return all block texture currently, may be expanded to include items later"""
        return self.get_all_block_textures()


    def get_all_block_textures(self)->list:
        print("Generating block texture list...")

        self._tex_names= {}
        self._tex_imgs = []

        _v =self.assetLoader.list_block_textures()
        # pprint(_v)
        for i, (k,v) in enumerate(_v.items()):
            # print(i)
            # print(k)
            # print(self.process_texture(v))
            self._tex_imgs.append(self.process_texture(v))
            self._tex_names[k]= i


        return self._tex_imgs


    def get_texture_index(self, name:str)-> int:
        # pprint(self._tex_names)
        if "t:" not in name:
             name = "{0}:{1}".format("minecraft", name)

        if not hasattr(self,"_tex_imgs"):
            raise ValueError("texture list has not been loaded yet.")
        # print(self._tex_names)
        return self._tex_names[name]


    def process_texture(self, texture:Image)->Image:
        texture = Image.frombytes("RGBA", (16,16), texture)
        h, w =texture.size #get images size
        if h != w:# check image is square if not (for example due to animated texture) crop shorter side
            texture = texture.crop((0,0,min(h,w),min(h,w)))

        texture = texture.resize((self.mc_texture_size, self.mc_texture_size), Image.BOX)
        h1, w1 = texture.size
        if h1 != w1 != self.mc_texture_size:

            # print("size error,{0} ".format(texture))
            texture.show()
        texture= texture.convert("RGBA")
        return texture

