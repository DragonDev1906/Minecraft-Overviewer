#    This file is part of the Minecraft Overviewer.
#
#    Minecraft Overviewer is free software: you can redistribute it and/or
#    modify it under the terms of the GNU General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or (at
#    your option) any later version.
#
#    Minecraft Overviewer is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
#    Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with the Overviewer.  If not, see <http://www.gnu.org/licenses/>.
import imp

import math
from pprint import pprint
from random import randint
import numpy
from PIL import Image, ImageEnhance, ImageOps, ImageDraw
import logging
import functools

from . import util, texturegen
import logging

from .asset_loader import AssetLoader, AssetLoaderException

logger = logging.getLogger()

# global variables to collate information in @material decorators
blockmap_generators = {}

known_blocks = set()
used_datas = set()
max_blockid = 23000
max_data = 0

transparent_blocks = set()
solid_blocks = set([22000])
fluid_blocks = set()
nospawn_blocks = set()
nodata_blocks = set()

IMG_N =0

# This is here for circular import reasons.
# Please don't ask, I choose to repress these memories.
# ... okay fine I'll tell you.
# Initialising the C extension requires access to the globals above.
# Due to the circular import, this wouldn't work, unless we reload the
# module in the C extension or just move the import below its dependencies.
from .c_overviewer import alpha_over




color_map = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray",
             "light_gray", "cyan", "purple", "blue", "brown", "green", "red", "black"]


##
## Textures object
##
class Textures(object):
    """An object that generates a set of block sprites to use while
    rendering. It accepts a background color, north direction, and
    local textures path.
    """
    def __init__(self, texturepath=None, bgcolor=(26, 26, 26, 0), northdirection=0):
        self.bgcolor = bgcolor
        self.rotation = northdirection
        self.find_file_local_path = texturepath
        self.assetLoader = AssetLoader(texturepath, (16,16))
        # not yet configurable
        self.texture_size = 24
        self.texture_dimensions = (self.texture_size, self.texture_size)
        
        # this is set in in generate()
        self.generated = False

        # see load_image_texture()
        # self.texture_cache = {}

        # once we find a jarfile that contains a texture, we cache the ZipFile object here

    
    ##
    ## pickle support
    ##
    
    def __getstate__(self):
        # we must get rid of the huge image lists, and other images
        attributes = self.__dict__.copy()
        pprint(attributes)
        for attr in ['blockmap', 'biome_grass_texture', 'watertexture', 'lavatexture', 'firetexture', 'portaltexture', 'lightcolor', 'grasscolor', 'foliagecolor', 'watercolor', 'texture_cache']:
            try:
                del attributes[attr]
            except KeyError:
                pass
        # attributes['assetLoader']['jars'] = OrderedDict()
        return attributes


    def __setstate__(self, attrs):
        # regenerate textures, if needed
        for attr, val in list(attrs.items()):
            setattr(self, attr, val)
        self.texture_cache = {}
        if self.generated:
            self.generate()
            #todo: remove this call to generate
    
    ##
    ## The big one: generate()
    ##
    def process_texture(self, texture)->Image:
        # texture = Image.frombytes("RGBA", (16,16), texture)
        h, w =texture.size #get images size
        if h != w:# check image is square if not (for example due to animated texture) crop shorter side
            texture = texture.crop((0,0,min(h,w),min(h,w)))
        texture = texture.resize((16, 16), Image.BOX)
        texture =texture.convert("RGBA")
        return texture


    def generate(self):
        # Make sure we have the foliage/grasscolor images available
        try:
            self.load_foliage_color()
            self.load_grass_color()
        except AssetLoaderException as e:
            logging.error(
                "Your system is missing either minecraft:colormap/foliage "
                "or minecraft:colormap/grass. Either complement your "
                "resource pack with these texture files, or install the vanilla Minecraft "
                "client to use as a fallback.")
            raise e
        
        # generate biome grass mask
        self.biome_grass_texture = self.build_block(self.process_texture(self.assetLoader.load_image(
        "minecraft:block/grass_block_top")), self.process_texture(self.assetLoader.load_image(
        "minecraft:block/grass_block_side_overlay")))

        # generate the block
        global blockmap_generators
        global known_blocks, used_datas
        global max_blockid, max_data

        # Get the maximum possible size when using automatic generation
        block_renderer = texturegen.BlockRenderer(self, start_block_id=22000)
        # auto_max_block_size, auto_max_data_size = block_renderer.get_max_size()
        # max_blockid = max(max_blockid, auto_max_block_size)
        # max_data = max(max_data, auto_max_data_size)

        # Create Image Array
        self.blockmap = [None] * max_blockid * max_data

        for (blockid, data), texgen in list(blockmap_generators.items()):
            tex = texgen(self, blockid, data)
            self.blockmap[blockid * max_data + data] = self.generate_texture_tuple(tex)

        for (blockid, data), img in list(block_renderer.iter_for_generate()):
            self.blockmap[blockid * max_data + data] = self.generate_texture_tuple(img)
            known_blocks.add(blockid)



        if self.texture_size != 24:
            # rescale biome grass
            self.biome_grass_texture = self.biome_grass_texture.resize(self.texture_dimensions, Image.ANTIALIAS)

            # rescale the rest
            for i, tex in enumerate(self.blockmap):
                if tex is None:
                    continue
                block = tex[0]
                scaled_block = block.resize(self.texture_dimensions, Image.ANTIALIAS)
                self.blockmap[i] = self.generate_texture_tuple(scaled_block)

        self.generated = True

    ##
    ## Helpers for opening textures
    ##




    # def load_image(self, filename):
    #     """Returns an image object"""
    #
    #     try:
    #         img = self.texture_cache[filename]
    #         if isinstance(img, Exception):  # Did we cache an exception?
    #             raise img                   # Okay then, raise it.
    #         return img
    #     except KeyError:
    #         pass
    #
    #     try:
    #         fileobj = self.find_file(filename)
    #     except (TextureException, IOError) as e:
    #         # We cache when our good friend find_file can't find
    #         # a texture, so that we do not repeatedly search for it.
    #         self.texture_cache[filename] = e
    #         raise e
    #     buffer = BytesIO(fileobj.read())
    #     img = Image.open(buffer).convert("RGBA")
    #     self.texture_cache[filename] = img
    #     return img



    def load_water(self):
        """Special-case function for loading water."""
        watertexture = getattr(self, "watertexture", None)
        if watertexture:
            return watertexture
        watertexture = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/water_still")))
        self.watertexture = watertexture
        return watertexture

    def load_lava(self):
        """Special-case function for loading lava."""
        lavatexture = getattr(self, "lavatexture", None)
        if lavatexture:
            return lavatexture
        lavatexture = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/lava_still")))
        self.lavatexture = lavatexture
        return lavatexture

    def load_fire(self):
        """Special-case function for loading fire."""
        firetexture = getattr(self, "firetexture", None)
        if firetexture:
            return firetexture
        fireNS = self.process_texture(self.assetLoader.load_image(("minecraft:block/fire_0")))
        fireEW = self.process_texture(self.assetLoader.load_image(("minecraft:block/fire_1")))
        firetexture = (fireNS, fireEW)
        self.firetexture = firetexture
        return firetexture

    def load_portal(self):
        """Special-case function for loading portal."""
        portaltexture = getattr(self, "portaltexture", None)
        if portaltexture:
            return portaltexture
        portaltexture = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/nether_portal")))
        self.portaltexture = portaltexture
        return portaltexture

    def load_light_color(self):
        """Helper function to load the light color texture."""
        if hasattr(self, "lightcolor"):
            return self.lightcolor
        try:
            lightcolor = list(self.assetLoader.load_image("light_normal").getdata())
        except Exception:
            logging.warning("Light color image could not be found.")
            lightcolor = None
        self.lightcolor = lightcolor
        return lightcolor

    def load_grass_color(self):
        """Helper function to load the grass color texture."""
        if not hasattr(self, "grasscolor"):
            self.grasscolor = list(self.assetLoader.load_image(
                "minecraft:colormap/grass").getdata())
        return self.grasscolor

    def load_foliage_color(self):
        """Helper function to load the foliage color texture."""
        if not hasattr(self, "foliagecolor"):
            self.foliagecolor = list(self.assetLoader.load_image(
                "minecraft:colormap/foliage").getdata())
        return self.foliagecolor

    #I guess "watercolor" is wrong. But I can't correct as my texture pack don't define water color.
    def load_water_color(self):
        """Helper function to load the water color texture."""
        if not hasattr(self, "watercolor"):
            self.watercolor = list(self.assetLoader.load_image("minecraft:block/watercolor").getdata())
        return self.watercolor

    def _split_terrain(self, terrain):
        """Builds and returns a length 256 array of each 16x16 chunk
        of texture.
        """
        textures = []
        (terrain_width, terrain_height) = terrain.size
        texture_resolution = terrain_width / 16
        for y in range(16):
            for x in range(16):
                left = x*texture_resolution
                upper = y*texture_resolution
                right = left+texture_resolution
                lower = upper+texture_resolution
                region = terrain.transform(
                          (16, 16),
                          Image.EXTENT,
                          (left,upper,right,lower),
                          Image.BICUBIC)
                textures.append(region)

        return textures

    ##
    ## Image Transformation Functions
    ##

    @staticmethod
    def transform_image_top(img):
        """Takes a PIL image and rotates it left 45 degrees and shrinks the y axis
        by a factor of 2. Returns the resulting image, which will be 24x12 pixels

        """

        # Resize to 17x17, since the diagonal is approximately 24 pixels, a nice
        # even number that can be split in half twice
        img = img.resize((17, 17), Image.ANTIALIAS)

        # Build the Affine transformation matrix for this perspective
        transform = numpy.matrix(numpy.identity(3))
        # Translate up and left, since rotations are about the origin
        transform *= numpy.matrix([[1,0,8.5],[0,1,8.5],[0,0,1]])
        # Rotate 45 degrees
        ratio = math.cos(math.pi/4)
        #transform *= numpy.matrix("[0.707,-0.707,0;0.707,0.707,0;0,0,1]")
        transform *= numpy.matrix([[ratio,-ratio,0],[ratio,ratio,0],[0,0,1]])
        # Translate back down and right
        transform *= numpy.matrix([[1,0,-12],[0,1,-12],[0,0,1]])
        # scale the image down by a factor of 2
        transform *= numpy.matrix("[1,0,0;0,2,0;0,0,1]")

        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((24,12), Image.AFFINE, transform)
        return newimg

    @staticmethod
    def transform_image_side(img):
        """Takes an image and shears it for the left side of the cube (reflect for
        the right side)"""

        # Size of the cube side before shear
        img = img.resize((12,12), Image.ANTIALIAS)

        # Apply shear
        transform = numpy.matrix(numpy.identity(3))
        transform *= numpy.matrix("[1,0,0;-0.5,1,0;0,0,1]")

        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((12,18), Image.AFFINE, transform)
        return newimg

    @staticmethod
    def transform_image_slope(img):
        """Takes an image and shears it in the shape of a slope going up
        in the -y direction (reflect for +x direction). Used for minetracks"""

        # Take the same size as trasform_image_side
        img = img.resize((12,12), Image.ANTIALIAS)

        # Apply shear
        transform = numpy.matrix(numpy.identity(3))
        transform *= numpy.matrix("[0.75,-0.5,3;0.25,0.5,-3;0,0,1]")
        transform = numpy.array(transform)[:2,:].ravel().tolist()

        newimg = img.transform((24,24), Image.AFFINE, transform)

        return newimg


    @staticmethod
    def transform_image_angle(img, angle):
        """Takes an image an shears it in arbitrary angle with the axis of
        rotation being vertical.

        WARNING! Don't use angle = pi/2 (or multiplies), it will return
        a blank image (or maybe garbage).

        NOTE: angle is in the image not in game, so for the left side of a
        block angle = 30 degree.
        """

        # Take the same size as trasform_image_side
        img = img.resize((12,12), Image.ANTIALIAS)

        # some values
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # function_x and function_y are used to keep the result image in the
        # same position, and constant_x and constant_y are the coordinates
        # for the center for angle = 0.
        constant_x = 6.
        constant_y = 6.
        function_x = 6.*(1-cos_angle)
        function_y = -6*sin_angle
        big_term = ( (sin_angle * (function_x + constant_x)) - cos_angle* (function_y + constant_y))/cos_angle

        # The numpy array is not really used, but is helpful to
        # see the matrix used for the transformation.
        transform = numpy.array([[1./cos_angle, 0, -(function_x + constant_x)/cos_angle],
                                 [-sin_angle/(cos_angle), 1., big_term ],
                                 [0, 0, 1.]])

        transform = tuple(transform[0]) + tuple(transform[1])

        newimg = img.transform((24,24), Image.AFFINE, transform)

        return newimg


    def build_block(self, top, side):
        """From a top texture and a side texture, build a block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        original_texture = top.copy()
        top = self.transform_image_top(top)

        if not side:
            alpha_over(img, top, (0,0), top)
            return img

        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        # Darken the sides slightly. These methods also affect the alpha layer,
        # so save them first (we don't want to "darken" the alpha layer making
        # the block transparent)
        # print(side.mode)
        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)

        alpha_over(img, top, (0,0), top)
        alpha_over(img, side, (0,6), side)
        alpha_over(img, otherside, (12,6), otherside)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the block perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))

        return img

    def build_slab_block(self, top, side, upper):
        """From a top texture and a side texture, build a slab block image.
        top and side should be 16x16 image objects. Returns a 24x24 image

        """
        # cut the side texture in half
        mask = side.crop((0,8,16,16))
        side = Image.new(side.mode, side.size, self.bgcolor)
        alpha_over(side, mask,(0,0,16,8), mask)

        # plain slab
        top = self.transform_image_top(top)
        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)

        # upside down slab
        delta = 0
        if upper:
            delta = 6

        img = Image.new("RGBA", (24,24), self.bgcolor)
        alpha_over(img, side, (0,12 - delta), side)
        alpha_over(img, otherside, (12,12 - delta), otherside)
        alpha_over(img, top, (0,6 - delta), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the block perfectly tessellate-able
        if upper:
            for x,y in [(3,4), (7,2), (11,0)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,17), (17,15), (21,13)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))
        else:
            for x,y in [(3,10), (7,8), (11,6)]:
                # Copy a pixel to x,y from x+1,y
                img.putpixel((x,y), img.getpixel((x+1,y)))
            for x,y in [(13,23), (17,21), (21,19)]:
                # Copy a pixel to x,y from x-1,y
                img.putpixel((x,y), img.getpixel((x-1,y)))

        return img

    def build_full_block(self, top, side1, side2, side3, side4, bottom=None):
        """From a top texture, a bottom texture and 4 different side textures,
        build a full block with four differnts faces. All images should be 16x16
        image objects. Returns a 24x24 image. Can be used to render any block.

        side1 is in the -y face of the cube     (top left, east)
        side2 is in the +x                      (top right, south)
        side3 is in the -x                      (bottom left, north)
        side4 is in the +y                      (bottom right, west)

        A non transparent block uses top, side 3 and side 4.

        If top is a tuple then first item is the top image and the second
        item is an increment (integer) from 0 to 16 (pixels in the
        original minecraft texture). This increment will be used to crop the
        side images and to paste the top image increment pixels lower, so if
        you use an increment of 8, it will draw a half-block.

        NOTE: this method uses the bottom of the texture image (as done in
        minecraft with beds and cackes)

        """

        increment = 0
        if isinstance(top, tuple):
            increment = int(round((top[1] / 16.)*12.)) # range increment in the block height in pixels (half texture size)
            crop_height = increment
            top = top[0]
            if side1 is not None:
                side1 = side1.copy()
                ImageDraw.Draw(side1).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side2 is not None:
                side2 = side2.copy()
                ImageDraw.Draw(side2).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side3 is not None:
                side3 = side3.copy()
                ImageDraw.Draw(side3).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))
            if side4 is not None:
                side4 = side4.copy()
                ImageDraw.Draw(side4).rectangle((0, 0,16,crop_height),outline=(0,0,0,0),fill=(0,0,0,0))

        img = Image.new("RGBA", (24,24), self.bgcolor)

        # first back sides
        if side1 is not None :
            side1 = self.transform_image_side(side1)
            side1 = side1.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side.
            sidealpha = side1.split()[3]
            side1 = ImageEnhance.Brightness(side1).enhance(0.9)
            side1.putalpha(sidealpha)

            alpha_over(img, side1, (0,0), side1)


        if side2 is not None :
            side2 = self.transform_image_side(side2)

            # Darken this side.
            sidealpha2 = side2.split()[3]
            side2 = ImageEnhance.Brightness(side2).enhance(0.8)
            side2.putalpha(sidealpha2)

            alpha_over(img, side2, (12,0), side2)

        if bottom is not None :
            bottom = self.transform_image_top(bottom)
            alpha_over(img, bottom, (0,12), bottom)

        # front sides
        if side3 is not None :
            side3 = self.transform_image_side(side3)

            # Darken this side
            sidealpha = side3.split()[3]
            side3 = ImageEnhance.Brightness(side3).enhance(0.9)
            side3.putalpha(sidealpha)

            alpha_over(img, side3, (0,6), side3)

        if side4 is not None :
            side4 = self.transform_image_side(side4)
            side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)

            # Darken this side
            sidealpha = side4.split()[3]
            side4 = ImageEnhance.Brightness(side4).enhance(0.8)
            side4.putalpha(sidealpha)

            alpha_over(img, side4, (12,6), side4)

        if top is not None :
            top = self.transform_image_top(top)
            alpha_over(img, top, (0, increment), top)

        # Manually touch up 6 pixels that leave a gap because of how the
        # shearing works out. This makes the block perfectly tessellate-able
        for x,y in [(13,23), (17,21), (21,19)]:
            # Copy a pixel to x,y from x-1,y
            img.putpixel((x,y), img.getpixel((x-1,y)))
        for x,y in [(3,4), (7,2), (11,0)]:
            # Copy a pixel to x,y from x+1,y
            img.putpixel((x,y), img.getpixel((x+1,y)))
        global IMG_N
        # IMG_N +=1
        # img.save("C:/Datafile/LSelter/Documents/Minecraft-Overviewer/test_conf/debug/"+ str(IMG_N) + ".png")
        return img

    def build_sprite(self, side):
        """From a side texture, create a sprite-like texture such as those used
        for spiderwebs or flowers."""
        img = Image.new("RGBA", (24,24), self.bgcolor)

        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        alpha_over(img, side, (6,3), side)
        alpha_over(img, otherside, (6,3), otherside)
        return img

    def build_billboard(self, tex):
        """From a texture, create a billboard-like texture such as
        those used for tall grass or melon stems.
        """
        img = Image.new("RGBA", (24,24), self.bgcolor)

        front = tex.resize((14, 12), Image.ANTIALIAS)
        alpha_over(img, front, (5,9))
        return img

    def generate_opaque_mask(self, img):
        """ Takes the alpha channel of the image and generates a mask
        (used for lighting the block) that deprecates values of alpha
        smallers than 50, and sets every other value to 255. """

        alpha = img.split()[3]
        return alpha.point(lambda a: int(min(a, 25.5) * 10))

    def tint_texture(self, im, c):
        # apparently converting to grayscale drops the alpha channel?
        i = ImageOps.colorize(ImageOps.grayscale(im), (0,0,0), c)
        i.putalpha(im.split()[3]); # copy the alpha band back in. assuming RGBA
        return i

    def generate_texture_tuple(self, img):
        """ This takes an image and returns the needed tuple for the
        blockmap array."""
        if img is None:
            return None
        return (img, self.generate_opaque_mask(img))

##
## The other big one: @material and associated framework
##

# the material registration decorator
def material(blockid=[], data=[0], **kwargs):
    # mapping from property name to the set to store them in
    properties = {"transparent" : transparent_blocks, "solid" : solid_blocks, "fluid" : fluid_blocks, "nospawn" : nospawn_blocks, "nodata" : nodata_blocks}

    # make sure blockid and data are iterable
    try:
        iter(blockid)
    except Exception:
        blockid = [blockid,]
    try:
        iter(data)
    except Exception:
        data = [data,]

    def inner_material(func):
        global blockmap_generators
        global max_data, max_blockid

        # create a wrapper function with a known signature
        @functools.wraps(func)
        def func_wrapper(texobj, blockid, data):
            return func(texobj, blockid, data)

        used_datas.update(data)
        if max(data) >= max_data:
            max_data = max(data) + 1

        for block in blockid:
            # set the property sets appropriately
            known_blocks.update([block])
            if block >= max_blockid:
                max_blockid = block + 1
            for prop in properties:
                try:
                    if block in kwargs.get(prop, []):
                        properties[prop].update([block])
                except TypeError:
                    if kwargs.get(prop, False):
                        properties[prop].update([block])

            # populate blockmap_generators with our function
            for d in data:
                blockmap_generators[(block, d)] = func_wrapper

        return func_wrapper
    return inner_material

# shortcut function for pure block, default to solid, nodata
def block(blockid=[], top_image=None, side_image=None, **kwargs):
    new_kwargs = {'solid' : True, 'nodata' : True}
    new_kwargs.update(kwargs)

    if top_image is None:
        raise ValueError("top_image was not provided")

    if side_image is None:
        side_image = top_image

    @material(blockid=blockid, **new_kwargs)
    def inner_block(self, unused_id, unused_data):
        return self.build_block(self.process_texture(self.assetLoader.load_image((top_image))), self.process_texture(
            self.assetLoader.load_image((side_image))))
    return inner_block

# shortcut function for sprite block, defaults to transparent, nodata
def sprite(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)

    if imagename is None:
        raise ValueError("imagename was not provided")

    @material(blockid=blockid, **new_kwargs)
    def inner_sprite(self, unused_id, unused_data):
        return self.build_sprite(self.process_texture(self.assetLoader.load_image((imagename))))
    return inner_sprite

# shortcut function for billboard block, defaults to transparent, nodata
def billboard(blockid=[], imagename=None, **kwargs):
    new_kwargs = {'transparent' : True, 'nodata' : True}
    new_kwargs.update(kwargs)

    if imagename is None:
        raise ValueError("imagename was not provided")

    @material(blockid=blockid, **new_kwargs)
    def inner_billboard(self, unused_id, unused_data):
        return self.build_billboard(self.process_texture(self.assetLoader.load_image((imagename))))
    return inner_billboard


##
## and finally: actual texture definitions
##

# stone
@material(blockid=1, data=list(range(7)), solid=True)
def stone(self, blockid, data):
    if data == 0: # regular old-school stone
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone")))
    elif data == 1: # granite
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/granite")))
    elif data == 2: # polished granite
        img = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_granite")))
    elif data == 3: # diorite
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/diorite")))
    elif data == 4: # polished diorite
        img = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_diorite")))
    elif data == 5: # andesite
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/andesite")))
    elif data == 6: # polished andesite
        img = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_andesite")))
    return self.build_block(img, img)

@material(blockid=2, data=list(range(11))+[0x10,], solid=True)
def grass(self, blockid, data):
    # 0x10 bit means SNOW
    side_img = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/grass_block_side")))
    if data & 0x10:
        side_img = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/grass_block_snow")))
    img = self.build_block(self.process_texture(self.assetLoader.load_image((
        "minecraft:block/grass_block_top"))), side_img)
    if not data & 0x10:
        alpha_over(img, self.biome_grass_texture, (0, 0), self.biome_grass_texture)
    return img

# dirt
@material(blockid=3, data=list(range(3)), solid=True)
def dirt_blocks(self, blockid, data):
    side_img = self.process_texture(self.assetLoader.load_image(("minecraft:block/dirt")))
    if data == 0: # normal
        img =  self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dirt"))) , side_img )
    if data == 1: # grassless
        img = self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dirt"))) , side_img )
    if data == 2: # podzol
        side_img = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/podzol_side")))
        img = self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/podzol_top"))) , side_img )
    return img

# cobblestone
block(blockid=4, top_image="minecraft:block/cobblestone")

# wooden planks
@material(blockid=5, data=list(range(6)), solid=True)
def wooden_planks(self, blockid, data):
    if data == 0: # normal
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/oak_planks"))), self.process_texture(self.assetLoader.load_image((
            "minecraft:block/oak_planks"))))
    if data == 1: # pine
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/spruce_planks"))),self.process_texture(self.assetLoader.load_image((
            "minecraft:block/spruce_planks"))))
    if data == 2: # birch
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/birch_planks"))),self.process_texture(self.assetLoader.load_image((
        "minecraft:block/birch_planks"))))
    if data == 3: # jungle wood
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/jungle_planks"))),self.process_texture(self.assetLoader.load_image((
            "minecraft:block/jungle_planks"))))
    if data == 4: # acacia
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/acacia_planks"))),self.process_texture(self.assetLoader.load_image((
            "minecraft:block/acacia_planks"))))
    if data == 5: # dark oak
        return self.build_block(self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dark_oak_planks"))),self.process_texture(
            self.assetLoader.load_image(("minecraft:block/dark_oak_planks"))))

@material(blockid=6, data=list(range(16)), transparent=True)
def saplings(self, blockid, data):
    # usual saplings
    tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_sapling")))

    if data & 0x3 == 1: # spruce sapling
        tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_sapling")))
    elif data & 0x3 == 2: # birch sapling
        tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_sapling")))
    elif data & 0x3 == 3: # jungle sapling
        tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_sapling")))
    elif data & 0x3 == 4: # acacia sapling
        tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_sapling")))
    elif data & 0x3 == 5: # dark oak/roofed oak/big oak sapling
        tex = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dark_oak_sapling")))
    return self.build_sprite(tex)

sprite(blockid=11385, imagename="minecraft:block/oak_sapling")
sprite(blockid=11386, imagename="minecraft:block/spruce_sapling")
sprite(blockid=11387, imagename="minecraft:block/birch_sapling")
sprite(blockid=11388, imagename="minecraft:block/jungle_sapling")
sprite(blockid=11389, imagename="minecraft:block/acacia_sapling")
sprite(blockid=11390, imagename="minecraft:block/dark_oak_sapling")
sprite(blockid=11413, imagename="minecraft:block/bamboo_stage0")

# bedrock
block(blockid=7, top_image="minecraft:block/bedrock")

# water, glass, and ice (no inner surfaces)
# uses pseudo-ancildata found in iterate.c
@material(blockid=[8, 9, 20, 79, 95], data=list(range(512)), fluid=(8, 9), transparent=True, nospawn=True, solid=(79, 20, 95))
def no_inner_surfaces(self, blockid, data):
    if blockid == 8 or blockid == 9:
        texture = self.load_water()
    # elif blockid == 20:
    #     texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/glass")))
    elif blockid == 95:
        texture = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/%s_stained_glass" % color_map[data & 0x0f])))
    else:
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/ice")))

    # now that we've used the lower 4 bits to get color, shift down to get the 5 bits that encode face hiding
    if not (blockid == 8 or blockid == 9): # water doesn't have a shifted pseudodata
        data = data >> 4

    if (data & 0b10000) == 16:
        top = texture
    else:
        top = None

    if (data & 0b0001) == 1:
        side1 = texture    # top left
    else:
        side1 = None

    if (data & 0b1000) == 8:
        side2 = texture    # top right
    else:
        side2 = None

    if (data & 0b0010) == 2:
        side3 = texture    # bottom left
    else:
        side3 = None

    if (data & 0b0100) == 4:
        side4 = texture    # bottom right
    else:
        side4 = None

    # if nothing shown do not draw at all
    if top is None and side3 is None and side4 is None:
        return None

    img = self.build_full_block(top,None,None,side3,side4)
    return img

@material(blockid=[10, 11], data=list(range(16)), fluid=True, transparent=False, nospawn=True)
def lava(self, blockid, data):
    lavatex = self.load_lava()
    return self.build_block(lavatex, lavatex)


# gravel
block(blockid=13, top_image="minecraft:block/gravel")
# gold ore
block(blockid=14, top_image="minecraft:block/gold_ore")
# iron ore
block(blockid=15, top_image="minecraft:block/iron_ore")
# coal ore
block(blockid=16, top_image="minecraft:block/coal_ore")

@material(blockid=[17,162,11306,11307,11308,11309,11310,11311], data=list(range(12)), solid=True)
def wood(self, blockid, data):
    # extract orientation and wood type frorm data bits
    wood_type = data & 3
    wood_orientation = data & 12
    if self.rotation == 1:
        if wood_orientation == 4: wood_orientation = 8
        elif wood_orientation == 8: wood_orientation = 4
    elif self.rotation == 3:
        if wood_orientation == 4: wood_orientation = 8
        elif wood_orientation == 8: wood_orientation = 4

    # choose textures
    if blockid == 17: # regular wood:
        if wood_type == 0: # normal
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_log")))
        if wood_type == 1: # spruce
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/spruce_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_log")))
        if wood_type == 2: # birch
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/birch_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_log")))
        if wood_type == 3: # jungle wood
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/jungle_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_log")))
    elif blockid == 162: # acacia/dark wood:
        if wood_type == 0: # acacia
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/acacia_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_log")))
        elif wood_type == 1: # dark oak
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/dark_oak_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/dark_oak_log")))
        else:
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/acacia_log_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_log")))
    if blockid == 11306: # stripped regular wood:
        if wood_type == 0: # normal
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_oak_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_oak_log")))
        if wood_type == 1: # spruce
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_spruce_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_spruce_log")))
        if wood_type == 2: # birch
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_birch_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_birch_log")))
        if wood_type == 3: # jungle wood
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_jungle_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_jungle_log")))
    elif blockid == 11307: # stripped acacia/dark wood:
        if wood_type == 0: # acacia
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log")))
        elif wood_type == 1: # dark oak
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_dark_oak_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_dark_oak_log")))
        else:
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log")))
    if blockid == 11308: # regular bark:
        if wood_type == 0: # normal
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_log")))
            side = top
        if wood_type == 1: # spruce
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_log")))
            side = top
        if wood_type == 2: # birch
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_log")))
            side = top
        if wood_type == 3: # jungle wood
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_log")))
            side = top
    elif blockid == 11309: # acacia/dark bark:
        if wood_type == 0: # acacia
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_log")))
            side = top
        elif wood_type == 1: # dark oak
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/dark_oak_log")))
            side = top
        else:
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_log")))
            side = top
    if blockid == 11310: # stripped regular wood:
        if wood_type == 0: # normal
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_oak_log")))
            side = top
        if wood_type == 1: # spruce
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_spruce_log")))
            side = top
        if wood_type == 2: # birch
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_birch_log")))
            side = top
        if wood_type == 3: # jungle wood
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_jungle_log")))
            side = top
    elif blockid == 11311: # stripped acacia/dark wood:
        if wood_type == 0: # acacia
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log")))
            side = top
        elif wood_type == 1: # dark oak
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_dark_oak_log")))
            side = top
        else:
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stripped_acacia_log")))
            side = top

    # choose orientation and paste textures
    if wood_orientation == 0:
        return self.build_block(top, side)
    elif wood_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif wood_orientation == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(270), top)

@material(blockid=[18, 161], data=list(range(16)), transparent=True, solid=True)
def leaves(self, blockid, data):
    # mask out the bits 4 and 8
    # they are used for player placed and check-for-decay block
    data = data & 0x7
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_leaves")))
    if (blockid, data) == (18, 1): # pine!
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_leaves")))
    elif (blockid, data) == (18, 2): # birth tree
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_leaves")))
    elif (blockid, data) == (18, 3): # jungle tree
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_leaves")))
    elif (blockid, data) == (161, 4): # acacia tree
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_leaves")))
    elif (blockid, data) == (161, 5):
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_leaves")))
    return self.build_block(t, t)

# sponge
block(blockid=19, top_image="minecraft:block/sponge")
# lapis lazuli ore
block(blockid=21, top_image="minecraft:block/lapis_ore")
# lapis lazuli block
block(blockid=22, top_image="minecraft:block/lapis_block")

# dispensers, dropper, furnaces, and burning furnaces
@material(blockid=[23, 61, 62, 158], data=list(range(6)), solid=True)
def furnaces(self, blockid, data):
    # first, do the rotation if needed
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/furnace_top")))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/furnace_side")))

    if blockid == 61:
        front = self.process_texture(self.assetLoader.load_image(("minecraft:block/furnace_front")))
    elif blockid == 62:
        front = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/furnace_front_on")))
    elif blockid == 23:
        front = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dispenser_front")))
        if data == 0: # dispenser pointing down
            return self.build_block(top, top)
        elif data == 1: # dispenser pointing up
            dispenser_top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/dispenser_front_vertical")))
            return self.build_block(dispenser_top, top)
    elif blockid == 158:
        front = self.process_texture(self.assetLoader.load_image(("minecraft:block/dropper_front")))
        if data == 0: # dropper pointing down
            return self.build_block(top, top)
        elif data == 1: # dispenser pointing up
            dropper_top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/dropper_front_vertical")))
            return self.build_block(dropper_top, top)

    if data == 3: # pointing west
        return self.build_full_block(top, None, None, side, front)
    elif data == 4: # pointing north
        return self.build_full_block(top, None, None, front, side)
    else: # in any other direction the front can't be seen
        return self.build_full_block(top, None, None, side, side)

# sandstone
@material(blockid=24, data=list(range(3)), solid=True)
def sandstone(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/sandstone_top")))
    if data == 0: # normal
        return self.build_block(top, self.process_texture(self.assetLoader.load_image((
            "minecraft:block/sandstone"))))
    if data == 1: # hieroglyphic
        return self.build_block(top, self.process_texture(self.assetLoader.load_image((
            "minecraft:block/chiseled_sandstone"))))
    if data == 2: # soft
        return self.build_block(top, self.process_texture(self.assetLoader.load_image((
            "minecraft:block/cut_sandstone"))))

# red sandstone
@material(blockid=179, data=list(range(3)), solid=True)
def sandstone(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/red_sandstone_top")))
    if data == 0: # normal
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone")))
            return self.build_full_block(top, None, None, side, side, self.process_texture(
                self.assetLoader.load_image(("minecraft:block/red_sandstone_bottom") )))
    if data == 1: # hieroglyphic
        return self.build_block(top, self.process_texture(self.assetLoader.load_image((
            "minecraft:block/chiseled_red_sandstone"))))
    if data == 2: # soft
        return self.build_block(top, self.process_texture(self.assetLoader.load_image((
            "minecraft:block/cut_red_sandstone"))))

# note block
block(blockid=25, top_image="minecraft:block/note_block")

@material(blockid=26, data=list(range(12)), transparent=True, nospawn=True)
def bed(self, blockid, data):
    # first get rotation done
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2

    bed_texture = self.assetLoader.load_image("minecraft:entity/bed/red") # FIXME: do tile entity
    # colours
    increment = 8
    left_face = None
    right_face = None
    top_face = None
    if data & 0x8 == 0x8: # head of the bed
        top = bed_texture.copy().crop((6,6,22,22))

        # Composing the side
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,6,6,22)).rotate(90, expand=True)
        # foot of the bed
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (0,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((6,0,22,6)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
            right_face = end
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            left_face = end
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            right_face = side

    else: # foot of the bed
        top = bed_texture.copy().crop((6,28,22,44))
        side = Image.new("RGBA", (16,16))
        side_part1 = bed_texture.copy().crop((0,28,6,44)).rotate(90, expand=True)
        side_part2 = bed_texture.copy().crop((53,3,56,6))
        side_part2_f = side_part2.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(side, side_part1, (0,7), side_part1)
        alpha_over(side, side_part2, (13,13), side_part2)

        end = Image.new("RGBA", (16,16))
        end_part = bed_texture.copy().crop((22,22,38,28)).rotate(180)
        alpha_over(end, end_part, (0,7), end_part)
        alpha_over(end, side_part2, (0,13), side_part2)
        alpha_over(end, side_part2_f, (13,13), side_part2_f)
        if data & 0x00 == 0x00: # South
            top_face = top.rotate(180)
            left_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x01 == 0x01: # West
            top_face = top.rotate(90)
            right_face = side.transpose(Image.FLIP_LEFT_RIGHT)
        if data & 0x02 == 0x02: # North
            top_face = top
            left_face = side
            right_face = end
        if data & 0x03 == 0x03: # East
            top_face = top.rotate(270)
            left_face = end
            right_face = side

    top_face = (top_face, increment)
    return self.build_full_block(top_face, None, None, left_face, right_face)

# powered, detector, activator and normal rails
@material(blockid=[27, 28, 66, 157], data=list(range(14)), transparent=True)
def rails(self, blockid, data):
    # first, do rotation
    # Masked to not clobber powered rail on/off info
    # Ascending and flat straight
    if self.rotation == 1:
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 0: data = data & 0b1000 | 1
        elif (data & 0b0111) == 1: data = data & 0b1000 | 0
        elif (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2
    if blockid == 66: # normal minetrack only
        #Corners
        if self.rotation == 1:
            if data == 6: data = 7
            elif data == 7: data = 8
            elif data == 8: data = 6
            elif data == 9: data = 9
        elif self.rotation == 2:
            if data == 6: data = 8
            elif data == 7: data = 9
            elif data == 8: data = 6
            elif data == 9: data = 7
        elif self.rotation == 3:
            if data == 6: data = 9
            elif data == 7: data = 6
            elif data == 8: data = 8
            elif data == 9: data = 7
    img = Image.new("RGBA", (24,24), self.bgcolor)

    if blockid == 27: # powered rail
        if data & 0x8 == 0: # unpowered
            raw_straight = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/powered_rail")))
            raw_corner = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/rail_corner")  ))  # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/powered_rail_on")))
            raw_corner = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/rail_corner") ))  # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7

    elif blockid == 28: # detector rail
        raw_straight = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/detector_rail")))
        raw_corner = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/rail_corner")))    # leave corners for code simplicity

    elif blockid == 66: # normal rail
        raw_straight = self.process_texture(self.assetLoader.load_image(("minecraft:block/rail")))
        raw_corner = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/rail_corner")))

    elif blockid == 157: # activator rail
        if data & 0x8 == 0: # unpowered
            raw_straight = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/activator_rail")))
            raw_corner = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/rail_corner")))    # they don't exist but make the code
                                                # much simplier
        elif data & 0x8 == 0x8: # powered
            raw_straight = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/activator_rail_on")))
            raw_corner = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/rail_corner")))    # leave corners for code simplicity
        # filter the 'powered' bit
        data = data & 0x7

    ## use transform_image to scale and shear
    if data == 0:
        track = self.transform_image_top(raw_straight)
        alpha_over(img, track, (0,12), track)
    elif data == 6:
        track = self.transform_image_top(raw_corner)
        alpha_over(img, track, (0,12), track)
    elif data == 7:
        track = self.transform_image_top(raw_corner.rotate(270))
        alpha_over(img, track, (0,12), track)
    elif data == 8:
        # flip
        track = self.transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM).rotate(90))
        alpha_over(img, track, (0,12), track)
    elif data == 9:
        track = self.transform_image_top(raw_corner.transpose(Image.FLIP_TOP_BOTTOM))
        alpha_over(img, track, (0,12), track)
    elif data == 1:
        track = self.transform_image_top(raw_straight.rotate(90))
        alpha_over(img, track, (0,12), track)

    #slopes
    elif data == 2: # slope going up in +x direction
        track = self.transform_image_slope(raw_straight)
        track = track.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, track, (2,0), track)
        # the 2 pixels move is needed to fit with the adjacent tracks

    elif data == 3: # slope going up in -x direction
        # tracks are sprites, in this case we are seeing the "side" of
        # the sprite, so draw a line to make it beautiful.
        ImageDraw.Draw(img).line([(11,11),(23,17)],fill=(164,164,164))
        # grey from track texture (exterior grey).
        # the track doesn't start from image corners, be carefull drawing the line!
    elif data == 4: # slope going up in -y direction
        track = self.transform_image_slope(raw_straight)
        alpha_over(img, track, (0,0), track)

    elif data == 5: # slope going up in +y direction
        # same as "data == 3"
        ImageDraw.Draw(img).line([(1,17),(12,11)],fill=(164,164,164))

    return img

# sticky and normal piston body
@material(blockid=[29, 33], data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, solid=True, nospawn=True)
def piston(self, blockid, data):
    # first, rotation
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2

    if blockid == 29: # sticky
        piston_t = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/piston_top_sticky"))).copy()
    else: # normal
        piston_t = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/piston_top"))).copy()

    # other textures
    side_t = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/piston_side"))).copy()
    back_t = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/piston_bottom"))).copy()
    interior_t = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/piston_inner"))).copy()

    if data & 0x08 == 0x08: # pushed out, non full block, tricky stuff
        # remove piston texture from piston body
        ImageDraw.Draw(side_t).rectangle((0, 0,16,3),outline=(0,0,0,0),fill=(0,0,0,0))

        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = self.build_full_block(back_t ,None ,None ,side_t, side_t)

        elif data & 0x07 == 0x1: # up
            img = self.build_full_block((interior_t, 4) ,None ,None ,side_t, side_t)

        elif data & 0x07 == 0x2: # east
            img = self.build_full_block(side_t , None, None ,side_t.rotate(90), back_t)

        elif data & 0x07 == 0x3: # west
            img = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), None)
            temp = self.transform_image_side(interior_t)
            temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
            alpha_over(img, temp, (9,5), temp)

        elif data & 0x07 == 0x4: # north
            img = self.build_full_block(side_t.rotate(90) ,None ,None , None, side_t.rotate(270))
            temp = self.transform_image_side(interior_t)
            alpha_over(img, temp, (3,5), temp)

        elif data & 0x07 == 0x5: # south
            img = self.build_full_block(side_t.rotate(270) ,None , None ,back_t, side_t.rotate(90))

    else: # pushed in, normal full block, easy stuff
        if data & 0x07 == 0x0: # down
            side_t = side_t.rotate(180)
            img = self.build_full_block(back_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x1: # up
            img = self.build_full_block(piston_t ,None ,None ,side_t, side_t)
        elif data & 0x07 == 0x2: # east
            img = self.build_full_block(side_t ,None ,None ,side_t.rotate(90), back_t)
        elif data & 0x07 == 0x3: # west
            img = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        elif data & 0x07 == 0x4: # north
            img = self.build_full_block(side_t.rotate(90) ,None ,None ,piston_t, side_t.rotate(270))
        elif data & 0x07 == 0x5: # south
            img = self.build_full_block(side_t.rotate(270) ,None ,None ,back_t, side_t.rotate(90))

    return img

# sticky and normal piston shaft
@material(blockid=34, data=[0,1,2,3,4,5,8,9,10,11,12,13], transparent=True, nospawn=True)
def piston_extension(self, blockid, data):
    # first, rotation
    # Masked to not clobber block head/foot info
    if self.rotation == 1:
        if (data & 0b0111) == 2: data = data & 0b1000 | 5
        elif (data & 0b0111) == 3: data = data & 0b1000 | 4
        elif (data & 0b0111) == 4: data = data & 0b1000 | 2
        elif (data & 0b0111) == 5: data = data & 0b1000 | 3
    elif self.rotation == 2:
        if (data & 0b0111) == 2: data = data & 0b1000 | 3
        elif (data & 0b0111) == 3: data = data & 0b1000 | 2
        elif (data & 0b0111) == 4: data = data & 0b1000 | 5
        elif (data & 0b0111) == 5: data = data & 0b1000 | 4
    elif self.rotation == 3:
        if (data & 0b0111) == 2: data = data & 0b1000 | 4
        elif (data & 0b0111) == 3: data = data & 0b1000 | 5
        elif (data & 0b0111) == 4: data = data & 0b1000 | 3
        elif (data & 0b0111) == 5: data = data & 0b1000 | 2

    if (data & 0x8) == 0x8: # sticky
        piston_t = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/piston_top_sticky"))).copy()
    else: # normal
        piston_t = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/piston_top"))).copy()

    # other textures
    side_t = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/piston_side"))).copy()
    back_t = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/piston_top"))).copy()
    # crop piston body
    ImageDraw.Draw(side_t).rectangle((0, 4,16,16),outline=(0,0,0,0),fill=(0,0,0,0))

    # generate the horizontal piston extension stick
    h_stick = Image.new("RGBA", (24,24), self.bgcolor)
    temp = self.transform_image_side(side_t)
    alpha_over(h_stick, temp, (1,7), temp)
    temp = self.transform_image_top(side_t.rotate(90))
    alpha_over(h_stick, temp, (1,1), temp)
    # Darken it
    sidealpha = h_stick.split()[3]
    h_stick = ImageEnhance.Brightness(h_stick).enhance(0.85)
    h_stick.putalpha(sidealpha)

    # generate the vertical piston extension stick
    v_stick = Image.new("RGBA", (24,24), self.bgcolor)
    temp = self.transform_image_side(side_t.rotate(90))
    alpha_over(v_stick, temp, (12,6), temp)
    temp = temp.transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over(v_stick, temp, (1,6), temp)
    # Darken it
    sidealpha = v_stick.split()[3]
    v_stick = ImageEnhance.Brightness(v_stick).enhance(0.85)
    v_stick.putalpha(sidealpha)

    # Piston orientation is stored in the 3 first bits
    if data & 0x07 == 0x0: # down
        side_t = side_t.rotate(180)
        img = self.build_full_block((back_t, 12) ,None ,None ,side_t, side_t)
        alpha_over(img, v_stick, (0,-3), v_stick)
    elif data & 0x07 == 0x1: # up
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(piston_t ,None ,None ,side_t, side_t)
        alpha_over(img, v_stick, (0,4), v_stick)
        alpha_over(img, img2, (0,0), img2)
    elif data & 0x07 == 0x2: # east
        img = self.build_full_block(side_t ,None ,None ,side_t.rotate(90), None)
        temp = self.transform_image_side(back_t).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, temp, (2,2), temp)
        alpha_over(img, h_stick, (6,3), h_stick)
    elif data & 0x07 == 0x3: # west
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(side_t.rotate(180) ,None ,None ,side_t.rotate(270), piston_t)
        alpha_over(img, h_stick, (0,0), h_stick)
        alpha_over(img, img2, (0,0), img2)
    elif data & 0x07 == 0x4: # north
        img = self.build_full_block(side_t.rotate(90) ,None ,None , piston_t, side_t.rotate(270))
        alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (0,0), h_stick.transpose(Image.FLIP_LEFT_RIGHT))
    elif data & 0x07 == 0x5: # south
        img = Image.new("RGBA", (24,24), self.bgcolor)
        img2 = self.build_full_block(side_t.rotate(270) ,None ,None ,None, side_t.rotate(90))
        temp = self.transform_image_side(back_t)
        alpha_over(img2, temp, (10,2), temp)
        alpha_over(img, img2, (0,0), img2)
        alpha_over(img, h_stick.transpose(Image.FLIP_LEFT_RIGHT), (-3,2), h_stick.transpose(Image.FLIP_LEFT_RIGHT))

    return img

# cobweb
sprite(blockid=30, imagename="minecraft:block/cobweb", nospawn=True)

@material(blockid=31, data=list(range(3)), transparent=True)
def tall_grass(self, blockid, data):
    if data == 0: # dead shrub
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/dead_bush")))
    elif data == 1: # tall grass
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/grass")))
    elif data == 2: # fern
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/fern")))

    return self.build_billboard(texture)

# dead bush
billboard(blockid=32, imagename="minecraft:block/dead_bush")

@material(blockid=35, data=list(range(16)), solid=True)
def wool(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_wool" %
                                                                color_map[data])))

    return self.build_block(texture, texture)

# dandelion
sprite(blockid=37, imagename="minecraft:block/dandelion")

# flowers
@material(blockid=38, data=list(range(10)), transparent=True)
def flower(self, blockid, data):
    flower_map = ["poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip", "orange_tulip",
                  "white_tulip", "pink_tulip", "oxeye_daisy", "dandelion"]
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s" %
                                                                flower_map[data])))

    return self.build_billboard(texture)

# brown mushroom
sprite(blockid=39, imagename="minecraft:block/brown_mushroom")
# red mushroom
sprite(blockid=40, imagename="minecraft:block/red_mushroom")
# block of gold
block(blockid=41, top_image="minecraft:block/gold_block")
# block of iron
block(blockid=42, top_image="minecraft:block/iron_block")

# double slabs and slabs
# these wooden slabs are unobtainable without cheating, they are still
# here because lots of pre-1.3 worlds use this block, add prismarine slabs
@material(blockid=[43, 44, 181, 182, 204, 205] + list(range(11340,11359)), data=list(range(16)),
          transparent=[44, 182, 205] + list(range(11340,11359)), solid=True)
def slabs(self, blockid, data):
    if blockid == 44 or blockid == 182:
        texture = data & 7
    else: # data > 8 are special double slabs
        texture = data

    if blockid == 44 or blockid == 43:
        if texture== 0: # stone slab
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone")))
        elif texture== 1: # sandstone slab
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/sandstone_top")))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/sandstone")))
        elif texture== 2: # wooden slab
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/oak_planks")))
        elif texture== 3: # cobblestone slab
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/cobblestone")))
        elif texture== 4: # brick
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/bricks")))
        elif texture== 5: # stone brick
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/stone_bricks")))
        elif texture== 6: # nether brick slab
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/nether_bricks")))
        elif texture== 7: #quartz
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/quartz_block_side")))
        elif texture== 8: # special stone double slab with top texture only
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/smooth_stone")))
        elif texture== 9: # special sandstone double slab with top texture only
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/sandstone_top")))
        else:
            return None

    elif blockid == 182: # single red sandstone slab
        if texture == 0:
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone")))
        else:
            return None

    elif blockid == 181: # double red sandstone slab
        if texture == 0: # red sandstone
            top = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone_top")))
            side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone")))
        elif texture == 8: # 'full' red sandstone (smooth)
            top = side = self.process_texture(self.assetLoader.load_image((
                "minecraft:block/red_sandstone_top")));
        else:
            return None
    elif blockid == 204 or blockid == 205: # purpur slab (single=205 double=204)
        top = side = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/purpur_block")));

    elif blockid == 11340: # prismarine slabs
        top = side = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/prismarine"))).copy()
    elif blockid == 11341: # dark prismarine slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/dark_prismarine"))).copy()
    elif blockid == 11342: #  prismarine brick slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/prismarine_bricks"))).copy()
    elif blockid == 11343: #  andesite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/andesite"))).copy()
    elif blockid == 11344: #  diorite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/diorite"))).copy()
    elif blockid == 11345: #  granite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/granite"))).copy()
    elif blockid == 11346: #  polished andesite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_andesite"))).copy()
    elif blockid == 11347: #  polished diorite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_diorite"))).copy()
    elif blockid == 11348: #  polished granite slabs
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/polished_granite"))).copy()
    elif blockid == 11349: #  red nether brick slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/red_nether_bricks"))).copy()
    elif blockid == 11350: #  smooth sandstone slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/sandstone_top"))).copy()
    elif blockid == 11351: #  cut sandstone slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/cut_sandstone"))).copy()
    elif blockid == 11352: #  smooth red sandstone slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/red_sandstone_top"))).copy()
    elif blockid == 11353: #  cut red sandstone slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/cut_red_sandstone"))).copy()
    elif blockid == 11354: #  end_stone_brick_slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/end_stone_bricks"))).copy()
    elif blockid == 11355: #  mossy_cobblestone_slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/mossy_cobblestone"))).copy()
    elif blockid == 11356: #  mossy_stone_brick_slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/mossy_stone_bricks"))).copy()
    elif blockid == 11357: #  smooth_quartz_slab
        top = side  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/quartz_block_bottom"))).copy()
    elif blockid == 11358: #  smooth_stone_slab
        top  = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/smooth_stone"))).copy()
        side = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/smooth_stone_slab_side"))).copy()

    if blockid == 43 or blockid == 181 or blockid == 204: # double slab
        return self.build_block(top, side)

    return self.build_slab_block(top, side, data & 8 == 8);

# brick block
block(blockid=45, top_image="minecraft:block/bricks")
# TNT
block(blockid=46, top_image="minecraft:block/tnt_top", side_image="minecraft:block/tnt_side", nospawn=True)
# bookshelf
block(blockid=47, top_image="minecraft:block/oak_planks", side_image="minecraft:block/bookshelf")
# moss stone
block(blockid=48, top_image="minecraft:block/mossy_cobblestone")
# obsidian
block(blockid=49, top_image="minecraft:block/obsidian")

# torch, redstone torch (off), redstone torch(on)
@material(blockid=[50, 75, 76], data=[1, 2, 3, 4, 5], transparent=True)
def torches(self, blockid, data):
    # first, rotations
    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2

    # choose the proper texture
    if blockid == 50: # torch
        small = self.process_texture(self.assetLoader.load_image(("minecraft:block/torch")))
    elif blockid == 75: # off redstone torch
        small = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/redstone_torch_off")))
    else: # on redstone torch
        small = self.process_texture(self.assetLoader.load_image((
            "minecraft:block/redstone_torch")))

    # compose a torch bigger than the normal
    # (better for doing transformations)
    torch = Image.new("RGBA", (16,16), self.bgcolor)
    alpha_over(torch,small,(-4,-3))
    alpha_over(torch,small,(-5,-2))
    alpha_over(torch,small,(-3,-2))

    # angle of inclination of the texture
    rotation = 15

    if data == 1: # pointing south
        torch = torch.rotate(-rotation, Image.NEAREST) # nearest filter is more nitid.
        img = self.build_full_block(None, None, None, torch, None, None)

    elif data == 2: # pointing north
        torch = torch.rotate(rotation, Image.NEAREST)
        img = self.build_full_block(None, None, torch, None, None, None)

    elif data == 3: # pointing west
        torch = torch.rotate(rotation, Image.NEAREST)
        img = self.build_full_block(None, torch, None, None, None, None)

    elif data == 4: # pointing east
        torch = torch.rotate(-rotation, Image.NEAREST)
        img = self.build_full_block(None, None, None, None, torch, None)

    elif data == 5: # standing on the floor
        # compose a "3d torch".
        img = Image.new("RGBA", (24,24), self.bgcolor)

        small_crop = small.crop((2,2,14,14))
        slice = small_crop.copy()
        ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))

        alpha_over(img, slice, (7,5))
        alpha_over(img, small_crop, (6,6))
        alpha_over(img, small_crop, (7,6))
        alpha_over(img, slice, (7,7))

    return img

# lantern
@material(blockid=11373, data=[0, 1], transparent=True)
def lantern(self, blockid, data):
    # get the  multipart texture of the lantern
    inputtexture = self.process_texture(self.assetLoader.load_image(("minecraft:block/lantern") ))

    # # now create a textures, using the parts defined in lantern.json

    # JSON data for sides
    # from": [ 5,  1,  5 ],
    #  "to": [11,  8, 11 ],
    # { "uv": [ 0, 2, 6,  9 ], "texture": "#all" }

    side_crop = inputtexture.crop((0, 2, 6, 9))
    side_slice = side_crop.copy()
    side_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    side_texture.paste(side_slice,(5, 8))

    # JSON data for top
    # { "uv": [  0, 9,  6, 15 ], "texture": "#all" }
    top_crop = inputtexture.crop((0, 9, 6, 15))
    top_slice = top_crop.copy()
    top_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    top_texture.paste(top_slice,(5, 5))

    # mimic parts of build_full_block, to get an object smaller than a block
    # build_full_block(self, top, side1, side2, side3, side4, bottom=None):
    # a non transparent block uses top, side 3 and side 4.
    img = Image.new("RGBA", (24, 24), self.bgcolor)
    # prepare the side textures
    # side3
    side3 = self.transform_image_side(side_texture)
    # Darken this side
    sidealpha = side3.split()[3]
    side3 = ImageEnhance.Brightness(side3).enhance(0.9)
    side3.putalpha(sidealpha)
    # place the transformed texture
    hangoff = 0
    if data == 1:
        hangoff = 8
    xoff = 4
    yoff =- hangoff
    alpha_over(img, side3, (xoff+0, yoff+6), side3)
    # side4
    side4 = self.transform_image_side(side_texture)
    side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)
    # Darken this side
    sidealpha = side4.split()[3]
    side4 = ImageEnhance.Brightness(side4).enhance(0.8)
    side4.putalpha(sidealpha)
    alpha_over(img, side4, (12-xoff, yoff+6), side4)
    # top
    top = self.transform_image_top(top_texture)
    alpha_over(img, top, (0, 8-hangoff), top)
    return img

# bamboo
@material(blockid=11416, transparent=True)
def bamboo(self, blockid, data):
    # get the  multipart texture of the lantern
    inputtexture = self.process_texture(self.assetLoader.load_image(("minecraft:block/bamboo_stalk") ))

    # # now create a textures, using the parts defined in bamboo1_age0.json
        # {   "from": [ 7, 0, 7 ],
        #     "to": [ 9, 16, 9 ],
        #     "faces": {
        #         "down":  { "uv": [ 13, 4, 15, 6 ], "texture": "#all", "cullface": "down" },
        #         "up":    { "uv": [ 13, 0, 15, 2], "texture": "#all", "cullface": "up" },
        #         "north": { "uv": [ 0, 0, 2, 16 ], "texture": "#all" },
        #         "south": { "uv": [ 0, 0, 2, 16 ], "texture": "#all" },
        #         "west":  { "uv": [  0, 0, 2, 16 ], "texture": "#all" },
        #         "east":  { "uv": [  0, 0, 2, 16 ], "texture": "#all" }
        #     }
        # }

    side_crop = inputtexture.crop((0, 0, 3, 16))
    side_slice = side_crop.copy()
    side_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    side_texture.paste(side_slice,(0, 0))

    # JSON data for top
    # "up":    { "uv": [ 13, 0, 15, 2], "texture": "#all", "cullface": "up" },
    top_crop = inputtexture.crop((13, 0, 16, 3))
    top_slice = top_crop.copy()
    top_texture = Image.new("RGBA", (16, 16), self.bgcolor)
    top_texture.paste(top_slice,(5, 5))

    # mimic parts of build_full_block, to get an object smaller than a block
    # build_full_block(self, top, side1, side2, side3, side4, bottom=None):
    # a non transparent block uses top, side 3 and side 4.
    img = Image.new("RGBA", (24, 24), self.bgcolor)
    # prepare the side textures
    # side3
    side3 = self.transform_image_side(side_texture)
    # Darken this side
    sidealpha = side3.split()[3]
    side3 = ImageEnhance.Brightness(side3).enhance(0.9)
    side3.putalpha(sidealpha)
    # place the transformed texture
    xoff = 3
    yoff = 0
    alpha_over(img, side3, (4+xoff, yoff), side3)
    # side4
    side4 = self.transform_image_side(side_texture)
    side4 = side4.transpose(Image.FLIP_LEFT_RIGHT)
    # Darken this side
    sidealpha = side4.split()[3]
    side4 = ImageEnhance.Brightness(side4).enhance(0.8)
    side4.putalpha(sidealpha)
    alpha_over(img, side4, (-4+xoff, yoff), side4)
    # top
    top = self.transform_image_top(top_texture)
    alpha_over(img, top, (-4+xoff, -5), top)
    return img

# composter
@material(blockid=11417, data=list(range(9)), transparent=True)
def composter(self, blockid, data):
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/composter_side") ))
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/composter_top") ))
    # bottom = self.process_texture(self.assetLoader.load_image(("minecraft:block/composter_bottom") ))

    if data == 0:  # empty
        return self.build_full_block(top, side, side, side, side)

    if data == 8:
        compost = self.transform_image_top(
            self.process_texture(self.assetLoader.load_image(("minecraft:block/composter_ready"))))
    else:
        compost = self.transform_image_top(
            self.process_texture(self.assetLoader.load_image((
                "minecraft:block/composter_compost"))))

    nudge = {1: (0, 9), 2: (0, 8), 3: (0, 7), 4: (0, 6), 5: (0, 4), 6: (0, 2), 7: (0, 0), 8: (0, 0)}

    img = self.build_full_block(None, side, side, None, None)
    alpha_over(img, compost, nudge[data], compost)
    img2 = self.build_full_block(top, None, None, side, side)
    alpha_over(img, img2, (0, 0), img2)
    return img

# fire
@material(blockid=51, data=list(range(16)), transparent=True)
def fire(self, blockid, data):
    firetextures = self.load_fire()
    side1 = self.transform_image_side(firetextures[0])
    side2 = self.transform_image_side(firetextures[1]).transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)

    alpha_over(img, side1, (12,0), side1)
    alpha_over(img, side2, (0,0), side2)

    alpha_over(img, side1, (0,6), side1)
    alpha_over(img, side2, (12,6), side2)

    return img

# monster spawner
block(blockid=52, top_image="minecraft:block/spawner", transparent=True)

# wooden, cobblestone, red brick, stone brick, netherbrick, sandstone, spruce, birch,
# jungle, quartz, red sandstone, (dark) prismarine, mossy brick and mossy cobblestone, stone smooth_quartz
# polished_granite polished_andesite polished_diorite granite diorite andesite end_stone_bricks red_nether_brick stairs
# smooth_red_sandstone_stairs
@material(blockid=[53, 67, 108, 109, 114, 128, 134, 135, 136, 156, 163, 164, 180, 203, 11337, 11338, 11339,
          11370, 11371, 11374, 11375, 11376, 11377, 11378, 11379, 11380, 11381, 11382, 11383, 11384, 11415],
          data=list(range(128)), transparent=True, solid=True, nospawn=True)
def stairs(self, blockid, data):
    # preserve the upside-down bit
    upside_down = data & 0x4

    # find solid quarters within the top or bottom half of the block
    #                   NW           NE           SE           SW
    quarters = [data & 0x8, data & 0x10, data & 0x20, data & 0x40]

    # rotate the quarters so we can pretend northdirection is always upper-left
    numpy.roll(quarters, [0,1,3,2][self.rotation])
    nw,ne,se,sw = quarters

    stair_id_to_tex = {
        53: "minecraft:block/oak_planks",
        67: "minecraft:block/cobblestone",
        108: "minecraft:block/bricks",
        109: "minecraft:block/stone_bricks",
        114: "minecraft:block/nether_bricks",
        128: "minecraft:block/sandstone",
        134: "minecraft:block/spruce_planks",
        135: "minecraft:block/birch_planks",
        136: "minecraft:block/jungle_planks",
        156: "minecraft:block/quartz_block_side",
        163: "minecraft:block/acacia_planks",
        164: "minecraft:block/dark_oak_planks",
        180: "minecraft:block/red_sandstone",
        203: "minecraft:block/purpur_block",
        11337: "minecraft:block/prismarine",
        11338: "minecraft:block/dark_prismarine",
        11339: "minecraft:block/prismarine_bricks",
        11370: "minecraft:block/mossy_stone_bricks",
        11371: "minecraft:block/mossy_cobblestone",
        11374: "minecraft:block/sandstone_top",
        11375: "minecraft:block/quartz_block_side",
        11376: "minecraft:block/polished_granite",
        11377: "minecraft:block/polished_diorite",
        11378: "minecraft:block/polished_andesite",
        11379: "minecraft:block/stone",
        11380: "minecraft:block/granite",
        11381: "minecraft:block/diorite",
        11382: "minecraft:block/andesite",
        11383: "minecraft:block/end_stone_bricks",
        11384: "minecraft:block/red_nether_bricks",
        11415: "minecraft:block/red_sandstone_top",
    }

    texture = self.process_texture(self.assetLoader.load_image((stair_id_to_tex[blockid]))).copy()

    outside_l = texture.copy()
    outside_r = texture.copy()
    inside_l = texture.copy()
    inside_r = texture.copy()

    # sandstone, red sandstone, and quartz stairs have special top texture
    special_tops = {
        128: "minecraft:block/sandstone_top",
        156: "minecraft:block/quartz_block_top",
        180: "minecraft:block/red_sandstone_top",
        11375: "minecraft:block/quartz_block_top",
    }

    if blockid in special_tops:
        texture = self.process_texture(self.assetLoader.load_image((special_tops[blockid]))).copy()


    slab_top = texture.copy()

    push = 8 if upside_down else 0

    def rect(tex,coords):
        ImageDraw.Draw(tex).rectangle(coords,outline=(0,0,0,0),fill=(0,0,0,0))

    # cut out top or bottom half from inner surfaces
    rect(inside_l, (0,8-push,15,15-push))
    rect(inside_r, (0,8-push,15,15-push))

    # cut out missing or obstructed quarters from each surface
    if not nw:
        rect(outside_l, (0,push,7,7+push))
        rect(texture, (0,0,7,7))
    if not nw or sw:
        rect(inside_r, (8,push,15,7+push)) # will be flipped
    if not ne:
        rect(texture, (8,0,15,7))
    if not ne or nw:
        rect(inside_l, (0,push,7,7+push))
    if not ne or se:
        rect(inside_r, (0,push,7,7+push)) # will be flipped
    if not se:
        rect(outside_r, (0,push,7,7+push)) # will be flipped
        rect(texture, (8,8,15,15))
    if not se or sw:
        rect(inside_l, (8,push,15,7+push))
    if not sw:
        rect(outside_l, (8,push,15,7+push))
        rect(outside_r, (8,push,15,7+push)) # will be flipped
        rect(texture, (0,8,7,15))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    if upside_down:
        # top should have no cut-outs after all
        texture = slab_top
    else:
        # render the slab-level surface
        slab_top = self.transform_image_top(slab_top)
        alpha_over(img, slab_top, (0,6))

    # render inner left surface
    inside_l = self.transform_image_side(inside_l)
    # Darken the vertical part of the second step
    sidealpha = inside_l.split()[3]
    # darken it a bit more than usual, looks better
    inside_l = ImageEnhance.Brightness(inside_l).enhance(0.8)
    inside_l.putalpha(sidealpha)
    alpha_over(img, inside_l, (6,3))

    # render inner right surface
    inside_r = self.transform_image_side(inside_r).transpose(Image.FLIP_LEFT_RIGHT)
    # Darken the vertical part of the second step
    sidealpha = inside_r.split()[3]
    # darken it a bit more than usual, looks better
    inside_r = ImageEnhance.Brightness(inside_r).enhance(0.7)
    inside_r.putalpha(sidealpha)
    alpha_over(img, inside_r, (6,3))

    # render outer surfaces
    alpha_over(img, self.build_full_block(texture, None, None, outside_l, outside_r))

    return img

# normal, locked (used in april's fool day), ender and trapped chest
# NOTE:  locked chest used to be id95 (which is now stained glass)
@material(blockid=[54, 130, 146], data=list(range(30)), transparent = True)
def chests(self, blockid, data):
    # the first 3 bits are the orientation as stored in minecraft,
    # bits 0x8 and 0x10 indicate which half of the double chest is it.

    # first, do the rotation if needed
    orientation_data = data & 7
    if self.rotation == 1:
        if orientation_data == 2: data = 5 | (data & 24)
        elif orientation_data == 3: data = 4 | (data & 24)
        elif orientation_data == 4: data = 2 | (data & 24)
        elif orientation_data == 5: data = 3 | (data & 24)
    elif self.rotation == 2:
        if orientation_data == 2: data = 3 | (data & 24)
        elif orientation_data == 3: data = 2 | (data & 24)
        elif orientation_data == 4: data = 5 | (data & 24)
        elif orientation_data == 5: data = 4 | (data & 24)
    elif self.rotation == 3:
        if orientation_data == 2: data = 4 | (data & 24)
        elif orientation_data == 3: data = 5 | (data & 24)
        elif orientation_data == 4: data = 3 | (data & 24)
        elif orientation_data == 5: data = 2 | (data & 24)

    if blockid == 130 and not data in [2, 3, 4, 5]: return None
        # iterate.c will only return the ancil data (without pseudo
        # ancil data) for locked and ender chests, so only
        # ancilData = 2,3,4,5 are used for this blockids

    if data & 24 == 0:
        if blockid == 130: t = self.assetLoader.load_image("minecraft:entity/chest/ender")
        else:
            try:
                t = self.assetLoader.load_image("minecraft:entity/chest/normal")
            except (AssetLoaderException, IOError):
                t = self.assetLoader.load_image("minecraft:entity/chest/chest")

        t = ImageOps.flip(t) # for some reason the 1.15 images are upside down

        # the textures is no longer in terrain.png, get it from
        # item/chest.png and get by cropping all the needed stuff
        if t.size != (64, 64): t = t.resize((64, 64), Image.ANTIALIAS)
        # top
        top = t.crop((28, 50, 42, 64))
        top.load() # every crop need a load, crop is a lazy operation
                   # see PIL manual
        img = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(img, top, (1, 1))
        top = img
        # front
        front_top = t.crop((42, 45, 56, 50))
        front_top.load()
        front_bottom = t.crop((42, 21, 56, 31))
        front_bottom.load()
        front_lock = t.crop((1, 59, 3, 63))
        front_lock.load()
        front = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(front, front_top, (1, 1))
        alpha_over(front, front_bottom, (1, 5))
        alpha_over(front, front_lock, (7, 3))
        # left side
        # left side, right side, and back are essentially the same for
        # the default texture, we take it anyway just in case other
        # textures make use of it.
        side_l_top = t.crop((14, 45, 28, 50))
        side_l_top.load()
        side_l_bottom = t.crop((14, 21, 28, 31))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_l, side_l_top, (1, 1))
        alpha_over(side_l, side_l_bottom, (1, 5))
        # right side
        side_r_top = t.crop((28, 45, 42, 50))
        side_r_top.load()
        side_r_bottom = t.crop((28, 21, 42, 31))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_r, side_r_top, (1, 1))
        alpha_over(side_r, side_r_bottom, (1, 5))
        # back
        back_top = t.crop((0, 45, 14, 50))
        back_top.load()
        back_bottom = t.crop((0, 21, 14, 31))
        back_bottom.load()
        back = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(back, back_top, (1, 1))
        alpha_over(back, back_bottom, (1, 5))

    else:
        # large chest
        # the textures is no longer in terrain.png, get it from
        # item/chest.png and get all the needed stuff
        t_left = self.assetLoader.load_image("minecraft:entity/chest/normal_left")
        t_right = self.assetLoader.load_image("minecraft:entity/chest/normal_right")
        # for some reason the 1.15 images are upside down
        t_left = ImageOps.flip(t_left)
        t_right = ImageOps.flip(t_right)

        # Top
        top_left = t_right.crop((29, 50, 44, 64))
        top_left.load()
        top_right = t_left.crop((29, 50, 44, 64))
        top_right.load()

        top = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(top,top_left, (1, 1))
        alpha_over(top,top_right, (16, 1))

        # Front
        front_top_left = t_left.crop((43, 45, 58, 50))
        front_top_left.load()
        front_top_right = t_right.crop((43, 45, 58, 50))
        front_top_right.load()

        front_bottom_left = t_left.crop((43, 21, 58, 31))
        front_bottom_left.load()
        front_bottom_right = t_right.crop((43, 21, 58, 31))
        front_bottom_right.load()

        front_lock = t_left.crop((1, 59, 3, 63))
        front_lock.load()

        front = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(front, front_top_left, (1, 1))
        alpha_over(front, front_top_right, (16, 1))
        alpha_over(front, front_bottom_left, (1, 5))
        alpha_over(front, front_bottom_right, (16, 5))
        alpha_over(front, front_lock, (15, 3))

        # Back
        back_top_left = t_right.crop((14, 45, 29, 50))
        back_top_left.load()
        back_top_right = t_left.crop((14, 45, 29, 50))
        back_top_right.load()

        back_bottom_left = t_right.crop((14, 21, 29, 31))
        back_bottom_left.load()
        back_bottom_right = t_left.crop((14, 21, 29, 31))
        back_bottom_right.load()

        back = Image.new("RGBA", (32, 16), self.bgcolor)
        alpha_over(back, back_top_left, (1, 1))
        alpha_over(back, back_top_right, (16, 1))
        alpha_over(back, back_bottom_left, (1, 5))
        alpha_over(back, back_bottom_right, (16, 5))

        # left side
        side_l_top = t_left.crop((29, 45, 43, 50))
        side_l_top.load()
        side_l_bottom = t_left.crop((29, 21, 43, 31))
        side_l_bottom.load()
        side_l = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_l, side_l_top, (1, 1))
        alpha_over(side_l, side_l_bottom, (1, 5))
        # right side
        side_r_top = t_right.crop((0, 45, 14, 50))
        side_r_top.load()
        side_r_bottom = t_right.crop((0, 21, 14, 31))
        side_r_bottom.load()
        side_r = Image.new("RGBA", (16, 16), self.bgcolor)
        alpha_over(side_r, side_r_top, (1, 1))
        alpha_over(side_r, side_r_bottom, (1, 5))

        if data & 24 == 8: # double chest, first half
            top = top.crop((0, 0, 16, 16))
            top.load()
            front = front.crop((0, 0, 16, 16))
            front.load()
            back = back.crop((0, 0, 16, 16))
            back.load()
            #~ side = side_l

        elif data & 24 == 16: # double, second half
            top = top.crop((16, 0, 32, 16))
            top.load()
            front = front.crop((16, 0, 32, 16))
            front.load()
            back = back.crop((16, 0, 32, 16))
            back.load()
            #~ side = side_r

        else: # just in case
            return None

    # compose the final block
    img = Image.new("RGBA", (24, 24), self.bgcolor)
    if data & 7 == 2: # north
        side = self.transform_image_side(side_r)
        alpha_over(img, side, (1, 7))
        back = self.transform_image_side(back)
        alpha_over(img, back.transpose(Image.FLIP_LEFT_RIGHT), (11, 7))
        front = self.transform_image_side(front)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0, 2))

    elif data & 7 == 3: # south
        side = self.transform_image_side(side_l)
        alpha_over(img, side, (1, 7))
        front = self.transform_image_side(front).transpose(Image.FLIP_LEFT_RIGHT)
        top = self.transform_image_top(top.rotate(180))
        alpha_over(img, top, (0, 2))
        alpha_over(img, front, (11, 7))

    elif data & 7 == 4: # west
        side = self.transform_image_side(side_r)
        alpha_over(img, side.transpose(Image.FLIP_LEFT_RIGHT), (11, 7))
        front = self.transform_image_side(front)
        alpha_over(img, front, (1, 7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0, 2))

    elif data & 7 == 5: # east
        back = self.transform_image_side(back)
        side = self.transform_image_side(side_l).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, side, (11, 7))
        alpha_over(img, back, (1, 7))
        top = self.transform_image_top(top.rotate(270))
        alpha_over(img, top, (0, 2))

    else: # just in case
        img = None

    return img

# redstone wire
# uses pseudo-ancildata found in iterate.c
@material(blockid=55, data=list(range(128)), transparent=True)
def wire(self, blockid, data):

    if data & 0b1000000 == 64: # powered redstone wire
        redstone_wire_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_dust_line0") )).rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(255,0,0))

        redstone_cross_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_dust_dot") ))
        redstone_cross_t = self.tint_texture(redstone_cross_t,(255,0,0))


    else: # unpowered redstone wire
        redstone_wire_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_dust_line0") )).rotate(90)
        redstone_wire_t = self.tint_texture(redstone_wire_t,(48,0,0))

        redstone_cross_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_dust_dot") ))
        redstone_cross_t = self.tint_texture(redstone_cross_t,(48,0,0))

    # generate an image per redstone direction
    branch_top_left = redstone_cross_t.copy()
    ImageDraw.Draw(branch_top_left).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    branch_top_right = redstone_cross_t.copy()
    ImageDraw.Draw(branch_top_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_top_right).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    branch_bottom_right = redstone_cross_t.copy()
    ImageDraw.Draw(branch_bottom_right).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_right).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_right).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    branch_bottom_left = redstone_cross_t.copy()
    ImageDraw.Draw(branch_bottom_left).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_left).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(branch_bottom_left).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # generate the bottom texture
    if data & 0b111111 == 0:
        bottom = redstone_cross_t.copy()

    # see iterate.c for where these masks come from
    has_x = (data & 0b1010) > 0
    has_z = (data & 0b0101) > 0
    if has_x and has_z:
        bottom = redstone_cross_t.copy()
        if has_x:
            alpha_over(bottom, redstone_wire_t.copy())
        if has_z:
            alpha_over(bottom, redstone_wire_t.copy().rotate(90))

    else:
        if has_x:
            bottom = redstone_wire_t.copy()
        elif has_z:
            bottom = redstone_wire_t.copy().rotate(90)
        elif data & 0b1111 == 0:
            bottom = redstone_cross_t.copy()

    # check for going up redstone wire
    if data & 0b100000 == 32:
        side1 = redstone_wire_t.rotate(90)
    else:
        side1 = None

    if data & 0b010000 == 16:
        side2 = redstone_wire_t.rotate(90)
    else:
        side2 = None

    img = self.build_full_block(None,side1,side2,None,None,bottom)

    return img

# diamond ore
block(blockid=56, top_image="minecraft:block/diamond_ore")
# diamond block
block(blockid=57, top_image="minecraft:block/diamond_block")

# crafting table
# needs two different sides
@material(blockid=58, solid=True, nodata=True)
def crafting_table(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/crafting_table_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/crafting_table_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/crafting_table_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

# fletching table
@material(blockid=11359, solid=True, nodata=True)
def fletching_table(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/fletching_table_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/fletching_table_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/fletching_table_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

# cartography table
@material(blockid=11360, solid=True, nodata=True)
def cartography_table(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/cartography_table_top") ))
    side1 = self.process_texture(self.assetLoader.load_image(("minecraft:block/cartography_table_side3") ))
    side2 = side1
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/cartography_table_side2") ))
    side4 = self.process_texture(self.assetLoader.load_image((
        "minecraft:block/cartography_table_side1"))).transpose(Image.FLIP_LEFT_RIGHT)

    img = self.build_full_block(top, side1, side2, side3, side4, None)
    return img

# smithing table
@material(blockid=11361, solid=True, nodata=True)
def smithing_table(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/smithing_table_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/smithing_table_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/smithing_table_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11362, solid=True, nodata=True)
def blast_furnace(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/blast_furnace_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/blast_furnace_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/blast_furnace_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11364, solid=True, nodata=True)
def smoker(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/smoker_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/smoker_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/smoker_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11366, solid=True, nodata=True)
def lectern(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/lectern_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/lectern_sides") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/lectern_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11367, solid=True, nodata=True)
def loom(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/loom_top") ))
    side3 = self.process_texture(self.assetLoader.load_image(("minecraft:block/loom_side") ))
    side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/loom_front") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11368, solid=True, nodata=True)
def stonecutter(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/stonecutter_top") ))
    side3 = side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/stonecutter_side") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img

@material(blockid=11369, solid=True, nodata=True)
def grindstone(self, blockid, data):
    top = side3 = side4 = self.process_texture(self.assetLoader.load_image(("minecraft:block/grindstone_side") ))

    img = self.build_full_block(top, None, None, side3, side4, None)
    return img


# crops with 8 data values (like wheat)
@material(blockid=59, data=list(range(8)), transparent=True, nospawn=True)
def crops8(self, blockid, data):
    raw_crop = self.process_texture(self.assetLoader.load_image(("minecraft:block/wheat_stage%d"
                                                                 % data)))
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# farmland and grass path (15/16 block)
@material(blockid=[60,208], data=list(range(9)), solid=True)
def farmland(self, blockid, data):
    if blockid == 60:
        side = self.process_texture(self.assetLoader.load_image(("minecraft:block/dirt") ))
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/farmland_moist") ))
        if data == 0:
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/farmland") ))
        # dirt.png is 16 pixels tall, so we need to crop it before building full block
        side = side.crop((0, 1, 16, 16))
        return self.build_full_block((top, 1), side, side, side, side)

    else:
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/grass_path_top") ))
        side = self.process_texture(self.assetLoader.load_image(("minecraft:block/grass_path_side") ))
        # side already has 1 transparent pixel at the top, so it doesn't need to be modified
        # just shift the top image down 1 pixel
        return self.build_full_block((top, 1), side, side, side, side)


# signposts
@material(blockid=[63,11401,11402,11403,11404,11405,11406], data=list(range(16)), transparent=True)
def signpost(self, blockid, data):

    # first rotations
    if self.rotation == 1:
        data = (data + 4) % 16
    elif self.rotation == 2:
        data = (data + 8) % 16
    elif self.rotation == 3:
        data = (data + 12) % 16

    sign_texture = {
        # (texture on sign, texture on stick)
        63: ("oak_planks.png", "oak_log.png"),
        11401: ("oak_planks.png", "oak_log.png"),
        11402: ("spruce_planks.png", "spruce_log.png"),
        11403: ("birch_planks.png", "birch_log.png"),
        11404: ("jungle_planks.png", "jungle_log.png"),
        11405: ("acacia_planks.png", "acacia_log.png"),
        11406: ("dark_oak_planks.png", "dark_oak_log.png"),
    }
    texture_path, texture_stick_path = ["assets/minecraft/textures/block/" + x for x in sign_texture[blockid]]

    texture = self.process_texture(self.assetLoader.load_image((texture_path))).copy()

    # cut the planks to the size of a signpost
    ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # If the signpost is looking directly to the image, draw some
    # random dots, they will look as text.
    if data in (0,1,2,3,4,5,15):
        for i in range(15):
            x = randint(4,11)
            y = randint(3,7)
            texture.putpixel((x,y),(0,0,0,255))

    # Minecraft uses wood texture for the signpost stick
    texture_stick = self.process_texture(self.assetLoader.load_image((texture_stick_path)))
    texture_stick = texture_stick.resize((12,12), Image.ANTIALIAS)
    ImageDraw.Draw(texture_stick).rectangle((2,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    #         W                N      ~90       E                   S        ~270
    angles = (330.,345.,0.,15.,30.,55.,95.,120.,150.,165.,180.,195.,210.,230.,265.,310.)
    angle = math.radians(angles[data])
    post = self.transform_image_angle(texture, angle)

    # choose the position of the "3D effect"
    incrementx = 0
    if data in (1,6,7,8,9,14):
        incrementx = -1
    elif data in (3,4,5,11,12,13):
        incrementx = +1

    alpha_over(img, texture_stick,(11, 8),texture_stick)
    # post2 is a brighter signpost pasted with a small shift,
    # gives to the signpost some 3D effect.
    post2 = ImageEnhance.Brightness(post).enhance(1.2)
    alpha_over(img, post2,(incrementx, -3),post2)
    alpha_over(img, post, (0,-2), post)

    return img


# wooden and iron door
# uses pseudo-ancildata found in iterate.c
@material(blockid=[64,71,193,194,195,196,197], data=list(range(32)), transparent=True)
def door(self, blockid, data):
    #Masked to not clobber block top/bottom & swung info
    if self.rotation == 1:
        if (data & 0b00011) == 0: data = data & 0b11100 | 1
        elif (data & 0b00011) == 1: data = data & 0b11100 | 2
        elif (data & 0b00011) == 2: data = data & 0b11100 | 3
        elif (data & 0b00011) == 3: data = data & 0b11100 | 0
    elif self.rotation == 2:
        if (data & 0b00011) == 0: data = data & 0b11100 | 2
        elif (data & 0b00011) == 1: data = data & 0b11100 | 3
        elif (data & 0b00011) == 2: data = data & 0b11100 | 0
        elif (data & 0b00011) == 3: data = data & 0b11100 | 1
    elif self.rotation == 3:
        if (data & 0b00011) == 0: data = data & 0b11100 | 3
        elif (data & 0b00011) == 1: data = data & 0b11100 | 0
        elif (data & 0b00011) == 2: data = data & 0b11100 | 1
        elif (data & 0b00011) == 3: data = data & 0b11100 | 2

    if data & 0x8 == 0x8: # top of the door
        if blockid == 64: # classic wood door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_door_top") ))
        elif blockid == 71: # iron door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/iron_door_top") ))
        elif blockid == 193: # spruce door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_door_top") ))
        elif blockid == 194: # birch door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_door_top") ))
        elif blockid == 195: # jungle door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_door_top") ))
        elif blockid == 196: # acacia door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_door_top") ))
        elif blockid == 197: # dark_oak door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_door_top") ))
    else: # bottom of the door
        if blockid == 64:
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_door_bottom") ))
        elif blockid == 71: # iron door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/iron_door_bottom") ))
        elif blockid == 193: # spruce door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_door_bottom") ))
        elif blockid == 194: # birch door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_door_bottom") ))
        elif blockid == 195: # jungle door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_door_bottom") ))
        elif blockid == 196: # acacia door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_door_bottom") ))
        elif blockid == 197: # dark_oak door
            raw_door = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_door_bottom") ))

    # if you want to render all doors as closed, then force
    # force closed to be True
    if data & 0x4 == 0x4:
        closed = False
    else:
        closed = True

    if data & 0x10 == 0x10:
        # hinge on the left (facing same door direction)
        hinge_on_left = True
    else:
        # hinge on the right (default single door)
        hinge_on_left = False

    # mask out the high bits to figure out the orientation
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if (data & 0x03) == 0: # facing west when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (0,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)

    if (data & 0x03) == 1: # facing north when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (0,6), tex)

        else:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (12,0), tex)


    if (data & 0x03) == 2: # facing east when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door)
                alpha_over(img, tex, (12,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door)
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (0,0), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (12,0), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)

    if (data & 0x03) == 3: # facing south when closed
        if hinge_on_left:
            if closed:
                tex = self.transform_image_side(raw_door).transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (12,0), tex)
        else:
            if closed:
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
                alpha_over(img, tex, (12,6), tex)
            else:
                # flip first to set the doornob on the correct side
                tex = self.transform_image_side(raw_door.transpose(Image.FLIP_LEFT_RIGHT))
                alpha_over(img, tex, (0,6), tex)

    return img

# ladder
@material(blockid=65, data=[2, 3, 4, 5], transparent=True)
def ladder(self, blockid, data):

    # first rotations
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    img = Image.new("RGBA", (24,24), self.bgcolor)
    raw_texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/ladder") ))

    if data == 5:
        # normally this ladder would be obsured by the block it's attached to
        # but since ladders can apparently be placed on transparent block, we
        # have to render this thing anyway.  same for data == 2
        tex = self.transform_image_side(raw_texture)
        alpha_over(img, tex, (0,6), tex)
        return img
    if data == 2:
        tex = self.transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, tex, (12,6), tex)
        return img
    if data == 3:
        tex = self.transform_image_side(raw_texture).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, tex, (0,0), tex)
        return img
    if data == 4:
        tex = self.transform_image_side(raw_texture)
        alpha_over(img, tex, (12,0), tex)
        return img


# wall signs
@material(blockid=[68,11407,11408,11409,11410,11411,11412], data=[2, 3, 4, 5], transparent=True)
def wall_sign(self, blockid, data): # wall sign

    # first rotations
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    sign_texture = {
        68: "oak_planks.png",
        11407: "oak_planks.png",
        11408: "spruce_planks.png",
        11409: "birch_planks.png",
        11410: "jungle_planks.png",
        11411: "acacia_planks.png",
        11412: "dark_oak_planks.png",
    }
    texture_path = "assets/minecraft/textures/block/" + sign_texture[blockid]
    texture = self.process_texture(self.assetLoader.load_image((texture_path) )).copy()
    # cut the planks to the size of a signpost
    ImageDraw.Draw(texture).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # draw some random black dots, they will look as text
    """ don't draw text at the moment, they are used in blank for decoration
    
    if data in (3,4):
        for i in range(15):
            x = randint(4,11)
            y = randint(3,7)
            texture.putpixel((x,y),(0,0,0,255))
    """

    img = Image.new("RGBA", (24,24), self.bgcolor)

    incrementx = 0
    if data == 2:  # east
        incrementx = +1
        sign = self.build_full_block(None, None, None, None, texture)
    elif data == 3:  # west
        incrementx = -1
        sign = self.build_full_block(None, texture, None, None, None)
    elif data == 4:  # north
        incrementx = +1
        sign = self.build_full_block(None, None, texture, None, None)
    elif data == 5:  # south
        incrementx = -1
        sign = self.build_full_block(None, None, None, texture, None)

    sign2 = ImageEnhance.Brightness(sign).enhance(1.2)
    alpha_over(img, sign2,(incrementx, 2),sign2)
    alpha_over(img, sign, (0,3), sign)

    return img

# levers
@material(blockid=69, data=list(range(16)), transparent=True)
def levers(self, blockid, data):
    if data & 8 == 8: powered = True
    else: powered = False

    data = data & 7

    # first rotations
    if self.rotation == 1:
        # on wall levers
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
        # on floor levers
        elif data == 5: data = 6
        elif data == 6: data = 5
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
        elif data == 5: data = 5
        elif data == 6: data = 6
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 5: data = 6
        elif data == 6: data = 5

    # generate the texture for the base of the lever
    t_base = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone") )).copy()

    ImageDraw.Draw(t_base).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t_base).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # generate the texture for the stick
    stick = self.process_texture(self.assetLoader.load_image(("minecraft:block/lever") )).copy()
    c_stick = Image.new("RGBA", (16,16), self.bgcolor)

    tmp = ImageEnhance.Brightness(stick).enhance(0.8)
    alpha_over(c_stick, tmp, (1,0), tmp)
    alpha_over(c_stick, stick, (0,0), stick)
    t_stick = self.transform_image_side(c_stick.rotate(45, Image.NEAREST))

    # where the lever will be composed
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # wall levers
    if data == 1: # facing SOUTH
        # levers can't be placed in transparent block, so this
        # direction is almost invisible
        return None

    elif data == 2: # facing NORTH
        base = self.transform_image_side(t_base)

        # paste it twice with different brightness to make a fake 3D effect
        alpha_over(img, base, (12,-1), base)

        alpha = base.split()[3]
        base = ImageEnhance.Brightness(base).enhance(0.9)
        base.putalpha(alpha)

        alpha_over(img, base, (11,0), base)

        # paste the lever stick
        pos = (7,-7)
        if powered:
            t_stick = t_stick.transpose(Image.FLIP_TOP_BOTTOM)
            pos = (7,6)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 3: # facing WEST
        base = self.transform_image_side(t_base)

        # paste it twice with different brightness to make a fake 3D effect
        base = base.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, base, (0,-1), base)

        alpha = base.split()[3]
        base = ImageEnhance.Brightness(base).enhance(0.9)
        base.putalpha(alpha)

        alpha_over(img, base, (1,0), base)

        # paste the lever stick
        t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
        pos = (5,-7)
        if powered:
            t_stick = t_stick.transpose(Image.FLIP_TOP_BOTTOM)
            pos = (6,6)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 4: # facing EAST
        # levers can't be placed in transparent block, so this
        # direction is almost invisible
        return None

    # floor levers
    elif data == 5: # pointing south when off
        # lever base, fake 3d again
        base = self.transform_image_top(t_base)

        alpha = base.split()[3]
        tmp = ImageEnhance.Brightness(base).enhance(0.8)
        tmp.putalpha(alpha)

        alpha_over(img, tmp, (0,12), tmp)
        alpha_over(img, base, (0,11), base)

        # lever stick
        pos = (3,2)
        if not powered:
            t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
            pos = (11,2)
        alpha_over(img, t_stick, pos, t_stick)

    elif data == 6: # pointing east when off
        # lever base, fake 3d again
        base = self.transform_image_top(t_base.rotate(90))

        alpha = base.split()[3]
        tmp = ImageEnhance.Brightness(base).enhance(0.8)
        tmp.putalpha(alpha)

        alpha_over(img, tmp, (0,12), tmp)
        alpha_over(img, base, (0,11), base)

        # lever stick
        pos = (2,3)
        if not powered:
            t_stick = t_stick.transpose(Image.FLIP_LEFT_RIGHT)
            pos = (10,2)
        alpha_over(img, t_stick, pos, t_stick)

    return img

# wooden and stone pressure plates, and weighted pressure plates
@material(blockid=[70, 72,147,148,11301,11302,11303,11304,11305], data=[0,1], transparent=True)
def pressure_plate(self, blockid, data):
    texture_name = {70:"minecraft:block/stone",              # stone
                    72:"minecraft:block/oak_planks",         # oak
                    11301:"minecraft:block/spruce_planks",   # spruce
                    11302:"minecraft:block/birch_planks",    # birch
                    11303:"minecraft:block/jungle_planks",   # jungle
                    11304:"minecraft:block/acacia_planks",   # acacia
                    11305:"minecraft:block/dark_oak_planks", # dark oak
                    147:"minecraft:block/gold_block",        # light golden
                    148:"minecraft:block/iron_block",        # heavy iron
                   }[blockid]
    t = self.process_texture(self.assetLoader.load_image((texture_name) )).copy()

    # cut out the outside border, pressure plates are smaller
    # than a normal block
    ImageDraw.Draw(t).rectangle((0,0,15,15),outline=(0,0,0,0))

    # create the textures and a darker version to make a 3d by
    # pasting them with an offstet of 1 pixel
    img = Image.new("RGBA", (24,24), self.bgcolor)

    top = self.transform_image_top(t)

    alpha = top.split()[3]
    topd = ImageEnhance.Brightness(top).enhance(0.8)
    topd.putalpha(alpha)

    #show it 3d or 2d if unpressed or pressed
    if data == 0:
        alpha_over(img,topd, (0,12),topd)
        alpha_over(img,top, (0,11),top)
    elif data == 1:
        alpha_over(img,top, (0,12),top)

    return img

# normal and glowing redstone ore
block(blockid=[73, 74], top_image="minecraft:block/redstone_ore")

# stone a wood buttons
@material(blockid=(77,143,11326,11327,11328,11329,11330), data=list(range(16)), transparent=True)
def buttons(self, blockid, data):

    # 0x8 is set if the button is pressed mask this info and render
    # it as unpressed
    data = data & 0x7

    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 4
        elif data == 3: data = 2
        elif data == 4: data = 1
        elif data == 5: data = 6
        elif data == 6: data = 5
    elif self.rotation == 2:
        if data == 1: data = 2
        elif data == 2: data = 1
        elif data == 3: data = 4
        elif data == 4: data = 3
    elif self.rotation == 3:
        if data == 1: data = 4
        elif data == 2: data = 3
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 5: data = 6
        elif data == 6: data = 5

    texturepath = {77:"minecraft:block/stone",
                   143:"minecraft:block/oak_planks",
                   11326:"minecraft:block/spruce_planks",
                   11327:"minecraft:block/birch_planks",
                   11328:"minecraft:block/jungle_planks",
                   11329:"minecraft:block/acacia_planks",
                   11330:"minecraft:block/dark_oak_planks"
                  }[blockid]
    t = self.process_texture(self.assetLoader.load_image((texturepath) )).copy()

    # generate the texture for the button
    ImageDraw.Draw(t).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(t).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    if data < 5:
        button = self.transform_image_side(t)

        if data == 1: # facing SOUTH
            # buttons can't be placed in transparent block, so this
            # direction can't be seen
            return None

        elif data == 2: # facing NORTH
            # paste it twice with different brightness to make a 3D effect
            alpha_over(img, button, (12,-1), button)

            alpha = button.split()[3]
            button = ImageEnhance.Brightness(button).enhance(0.9)
            button.putalpha(alpha)

            alpha_over(img, button, (11,0), button)

        elif data == 3: # facing WEST
            # paste it twice with different brightness to make a 3D effect
            button = button.transpose(Image.FLIP_LEFT_RIGHT)
            alpha_over(img, button, (0,-1), button)

            alpha = button.split()[3]
            button = ImageEnhance.Brightness(button).enhance(0.9)
            button.putalpha(alpha)

            alpha_over(img, button, (1,0), button)

        elif data == 4: # facing EAST
            # buttons can't be placed in transparent block, so this
            # direction can't be seen
            return None

    else:
        if data == 5: # long axis east-west
            button = self.transform_image_top(t)
        else: # long axis north-south
            button = self.transform_image_top(t.rotate(90))

        # paste it twice with different brightness to make a 3D effect
        alpha_over(img, button, (0,12), button)

        alpha = button.split()[3]
        button = ImageEnhance.Brightness(button).enhance(0.9)
        button.putalpha(alpha)

        alpha_over(img, button, (0,11), button)

    return img

# snow
@material(blockid=78, data=list(range(16)), transparent=True, solid=True)
def snow(self, blockid, data):
    # still not rendered correctly: data other than 0

    tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/snow") ))

    # make the side image, top 3/4 transparent
    mask = tex.crop((0,12,16,16))
    sidetex = Image.new(tex.mode, tex.size, self.bgcolor)
    alpha_over(sidetex, mask, (0,12,16,16), mask)

    img = Image.new("RGBA", (24,24), self.bgcolor)

    top = self.transform_image_top(tex)
    side = self.transform_image_side(sidetex)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    alpha_over(img, side, (0,6), side)
    alpha_over(img, otherside, (12,6), otherside)
    alpha_over(img, top, (0,9), top)

    return img

# snow block
block(blockid=80, top_image="minecraft:block/snow")

# cactus
@material(blockid=81, data=list(range(15)), transparent=True, solid=True, nospawn=True)
def cactus(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/cactus_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/cactus_side") ))

    img = Image.new("RGBA", (24,24), self.bgcolor)

    top = self.transform_image_top(top)
    side = self.transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)

    alpha_over(img, side, (1,6), side)
    alpha_over(img, otherside, (11,6), otherside)
    alpha_over(img, top, (0,0), top)

    return img

# clay block
block(blockid=82, top_image="minecraft:block/clay")

# sugar cane
@material(blockid=83, data=list(range(16)), transparent=True)
def sugar_cane(self, blockid, data):
    tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/sugar_cane") ))
    return self.build_sprite(tex)

# jukebox
@material(blockid=84, data=list(range(16)), solid=True)
def jukebox(self, blockid, data):
    return self.build_block(self.process_texture(self.assetLoader.load_image(("minecraft:block/jukebox_top") )), self.process_texture(self.assetLoader.load_image(("minecraft:block/note_block") )))

# nether and normal fences
# uses pseudo-ancildata found in iterate.c
@material(blockid=[85, 188, 189, 190, 191, 192, 113], data=list(range(16)), transparent=True, nospawn=True)
def fence(self, blockid, data):
    # no need for rotations, it uses pseudo data.
    # create needed images for Big stick fence
    if blockid == 85: # normal fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_planks") )).copy()
    elif blockid == 188: # spruce fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_planks") )).copy()
    elif blockid == 189: # birch fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_planks") )).copy()
    elif blockid == 190: # jungle fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_planks") )).copy()
    elif blockid == 191: # big/dark oak fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_planks") )).copy()
    elif blockid == 192: # acacia oak fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_planks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_planks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_planks") )).copy()
    else: # netherbrick fence
        fence_top = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_bricks") )).copy()
        fence_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_bricks") )).copy()
        fence_small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_bricks") )).copy()

    # generate the textures of the fence
    ImageDraw.Draw(fence_top).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,0,15,5),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_top).rectangle((0,10,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(fence_side).rectangle((0,0,5,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_side).rectangle((10,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the big stick
    fence_side = self.transform_image_side(fence_side)
    fence_other_side = fence_side.transpose(Image.FLIP_LEFT_RIGHT)
    fence_top = self.transform_image_top(fence_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = fence_side.split()[3]
    fence_side = ImageEnhance.Brightness(fence_side).enhance(0.9)
    fence_side.putalpha(sidealpha)
    othersidealpha = fence_other_side.split()[3]
    fence_other_side = ImageEnhance.Brightness(fence_other_side).enhance(0.8)
    fence_other_side.putalpha(othersidealpha)

    # Compose the fence big stick
    fence_big = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(fence_big,fence_side, (5,4),fence_side)
    alpha_over(fence_big,fence_other_side, (7,4),fence_other_side)
    alpha_over(fence_big,fence_top, (0,0),fence_top)

    # Now render the small sticks.
    # Create needed images
    ImageDraw.Draw(fence_small_side).rectangle((0,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,4,15,6),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(fence_small_side).rectangle((11,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the small sticks
    fence_small_side = self.transform_image_side(fence_small_side)
    fence_small_other_side = fence_small_side.transpose(Image.FLIP_LEFT_RIGHT)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = fence_small_other_side.split()[3]
    fence_small_other_side = ImageEnhance.Brightness(fence_small_other_side).enhance(0.9)
    fence_small_other_side.putalpha(sidealpha)
    sidealpha = fence_small_side.split()[3]
    fence_small_side = ImageEnhance.Brightness(fence_small_side).enhance(0.9)
    fence_small_side.putalpha(sidealpha)

    # Create img to compose the fence
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # Position of fence small sticks in img.
    # These postitions are strange because the small sticks of the
    # fence are at the very left and at the very right of the 16x16 images
    pos_top_left = (2,3)
    pos_top_right = (10,3)
    pos_bottom_right = (10,7)
    pos_bottom_left = (2,7)

    # +x axis points top right direction
    # +y axis points bottom right direction
    # First compose small sticks in the back of the image,
    # then big stick and thecn small sticks in the front.

    if (data & 0b0001) == 1:
        alpha_over(img,fence_small_side, pos_top_left,fence_small_side)                # top left
    if (data & 0b1000) == 8:
        alpha_over(img,fence_small_other_side, pos_top_right,fence_small_other_side)    # top right

    alpha_over(img,fence_big,(0,0),fence_big)

    if (data & 0b0010) == 2:
        alpha_over(img,fence_small_other_side, pos_bottom_left,fence_small_other_side)      # bottom left
    if (data & 0b0100) == 4:
        alpha_over(img,fence_small_side, pos_bottom_right,fence_small_side)                  # bottom right

    return img

# pumpkin
@material(blockid=[86, 91,11300], data=list(range(4)), solid=True)
def pumpkin(self, blockid, data): # pumpkins, jack-o-lantern
    # rotation
    if self.rotation == 1:
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif self.rotation == 2:
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif self.rotation == 3:
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2

    # texture generation
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/pumpkin_top") ))
    frontName = {86: "minecraft:block/pumpkin_side",
                 91: "minecraft:block/jack_o_lantern",
                 11300: "minecraft:block/carved_pumpkin"
                }[blockid]
    front = self.process_texture(self.assetLoader.load_image((frontName) ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/pumpkin_side") ))

    if data == 0: # pointing west
        img = self.build_full_block(top, None, None, side, front)

    elif data == 1: # pointing north
        img = self.build_full_block(top, None, None, front, side)

    else: # in any other direction the front can't be seen
        img = self.build_full_block(top, None, None, side, side)

    return img

# netherrack
block(blockid=87, top_image="minecraft:block/netherrack")

# soul sand
block(blockid=88, top_image="minecraft:block/soul_sand")

# glowstone
block(blockid=89, top_image="minecraft:block/glowstone")

# portal
@material(blockid=90, data=[1, 2, 4, 5, 8, 10], transparent=True)
def portal(self, blockid, data):
    # no rotations, uses pseudo data
    portaltexture = self.load_portal()
    img = Image.new("RGBA", (24,24), self.bgcolor)

    side = self.transform_image_side(portaltexture)
    otherside = side.transpose(Image.FLIP_TOP_BOTTOM)

    if data in (1,4,5):
        alpha_over(img, side, (5,4), side)

    if data in (2,8,10):
        alpha_over(img, otherside, (5,4), otherside)

    return img

# cake!
@material(blockid=92, data=list(range(6)), transparent=True, nospawn=True)
def cake(self, blockid, data):

    # cake textures
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/cake_top") )).copy()
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/cake_side") )).copy()
    fullside = side.copy()
    inside = self.process_texture(self.assetLoader.load_image(("minecraft:block/cake_inner") ))

    img = Image.new("RGBA", (24,24), self.bgcolor)
    if data == 0: # unbitten cake
        top = self.transform_image_top(top)
        side = self.transform_image_side(side)
        otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

        # darken sides slightly
        sidealpha = side.split()[3]
        side = ImageEnhance.Brightness(side).enhance(0.9)
        side.putalpha(sidealpha)
        othersidealpha = otherside.split()[3]
        otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
        otherside.putalpha(othersidealpha)

        # composite the cake
        alpha_over(img, side, (1,6), side)
        alpha_over(img, otherside, (11,7), otherside) # workaround, fixes a hole
        alpha_over(img, otherside, (12,6), otherside)
        alpha_over(img, top, (0,6), top)

    else:
        # cut the textures for a bitten cake
        coord = int(16./6.*data)
        ImageDraw.Draw(side).rectangle((16 - coord,0,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(top).rectangle((0,0,coord,16),outline=(0,0,0,0),fill=(0,0,0,0))

        # the bitten part of the cake always points to the west
        # composite the cake for every north orientation
        if self.rotation == 0: # north top-left
            # create right side
            rs = self.transform_image_side(side).transpose(Image.FLIP_LEFT_RIGHT)
            # create bitten side and its coords
            deltax = 2*data
            deltay = -1*data
            if data == 3: deltax += 1 # special case fixing pixel holes
            ls = self.transform_image_side(inside)
            # create top side
            t = self.transform_image_top(top)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, rs, (12,6), rs)
            alpha_over(img, ls, (1 + deltax,6 + deltay), ls)
            alpha_over(img, t, (0,6), t)

        elif self.rotation == 1: # north top-right
            # bitten side not shown
            # create left side
            ls = self.transform_image_side(side.transpose(Image.FLIP_LEFT_RIGHT))
            # create top
            t = self.transform_image_top(top.rotate(-90))
            # create right side
            rs = self.transform_image_side(fullside).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (0,6), t)
            alpha_over(img, rs, (12,6), rs)

        elif self.rotation == 2: # north bottom-right
            # bitten side not shown
            # left side
            ls = self.transform_image_side(fullside)
            # top
            t = self.transform_image_top(top.rotate(180))
            # right side
            rs = self.transform_image_side(side.transpose(Image.FLIP_LEFT_RIGHT)).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (1,6), t)
            alpha_over(img, rs, (12,6), rs)

        elif self.rotation == 3: # north bottom-left
            # create left side
            ls = self.transform_image_side(side)
            # create top
            t = self.transform_image_top(top.rotate(90))
            # create right side and its coords
            deltax = 12-2*data
            deltay = -1*data
            if data == 3: deltax += -1 # special case fixing pixel holes
            rs = self.transform_image_side(inside).transpose(Image.FLIP_LEFT_RIGHT)
            # darken sides slightly
            sidealpha = ls.split()[3]
            ls = ImageEnhance.Brightness(ls).enhance(0.9)
            ls.putalpha(sidealpha)
            othersidealpha = rs.split()[3]
            rs = ImageEnhance.Brightness(rs).enhance(0.8)
            rs.putalpha(othersidealpha)
            # compose the cake
            alpha_over(img, ls, (2,6), ls)
            alpha_over(img, t, (1,6), t)
            alpha_over(img, rs, (1 + deltax,6 + deltay), rs)

    return img

# redstone repeaters ON and OFF
@material(blockid=[93,94], data=list(range(16)), transparent=True, nospawn=True)
def repeater(self, blockid, data):
    # rotation
    # Masked to not clobber delay info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2

    # generate the diode
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/repeater") )) if blockid == 93 else self.process_texture(self.assetLoader.load_image(("minecraft:block/repeater_on") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/smooth_stone_slab_side") ))
    increment = 13

    if (data & 0x3) == 0: # pointing east
        pass

    if (data & 0x3) == 1: # pointing south
        top = top.rotate(270)

    if (data & 0x3) == 2: # pointing west
        top = top.rotate(180)

    if (data & 0x3) == 3: # pointing north
        top = top.rotate(90)

    img = self.build_full_block( (top, increment), None, None, side, side)

    # compose a "3d" redstone torch
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_torch_off") )).copy() if blockid == 93 else self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_torch") )).copy()
    torch = Image.new("RGBA", (24,24), self.bgcolor)

    t_crop = t.crop((2,2,14,14))
    slice = t_crop.copy()
    ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))

    alpha_over(torch, slice, (6,4))
    alpha_over(torch, t_crop, (5,5))
    alpha_over(torch, t_crop, (6,5))
    alpha_over(torch, slice, (6,6))

    # paste redstone torches everywhere!
    # the torch is too tall for the repeater, crop the bottom.
    ImageDraw.Draw(torch).rectangle((0,16,24,24),outline=(0,0,0,0),fill=(0,0,0,0))

    # touch up the 3d effect with big rectangles, just in case, for other texture packs
    ImageDraw.Draw(torch).rectangle((0,24,10,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(torch).rectangle((12,15,24,24),outline=(0,0,0,0),fill=(0,0,0,0))

    # torch positions for every redstone torch orientation.
    #
    # This is a horrible list of torch orientations. I tried to
    # obtain these orientations by rotating the positions for one
    # orientation, but pixel rounding is horrible and messes the
    # torches.

    if (data & 0x3) == 0: # pointing east
        if (data & 0xC) == 0: # one tick delay
            moving_torch = (1,1)
            static_torch = (-3,-1)

        elif (data & 0xC) == 4: # two ticks delay
            moving_torch = (2,2)
            static_torch = (-3,-1)

        elif (data & 0xC) == 8: # three ticks delay
            moving_torch = (3,2)
            static_torch = (-3,-1)

        elif (data & 0xC) == 12: # four ticks delay
            moving_torch = (4,3)
            static_torch = (-3,-1)

    elif (data & 0x3) == 1: # pointing south
        if (data & 0xC) == 0: # one tick delay
            moving_torch = (1,1)
            static_torch = (5,-1)

        elif (data & 0xC) == 4: # two ticks delay
            moving_torch = (0,2)
            static_torch = (5,-1)

        elif (data & 0xC) == 8: # three ticks delay
            moving_torch = (-1,2)
            static_torch = (5,-1)

        elif (data & 0xC) == 12: # four ticks delay
            moving_torch = (-2,3)
            static_torch = (5,-1)

    elif (data & 0x3) == 2: # pointing west
        if (data & 0xC) == 0: # one tick delay
            moving_torch = (1,1)
            static_torch = (5,3)

        elif (data & 0xC) == 4: # two ticks delay
            moving_torch = (0,0)
            static_torch = (5,3)

        elif (data & 0xC) == 8: # three ticks delay
            moving_torch = (-1,0)
            static_torch = (5,3)

        elif (data & 0xC) == 12: # four ticks delay
            moving_torch = (-2,-1)
            static_torch = (5,3)

    elif (data & 0x3) == 3: # pointing north
        if (data & 0xC) == 0: # one tick delay
            moving_torch = (1,1)
            static_torch = (-3,3)

        elif (data & 0xC) == 4: # two ticks delay
            moving_torch = (2,0)
            static_torch = (-3,3)

        elif (data & 0xC) == 8: # three ticks delay
            moving_torch = (3,0)
            static_torch = (-3,3)

        elif (data & 0xC) == 12: # four ticks delay
            moving_torch = (4,-1)
            static_torch = (-3,3)

    # this paste order it's ok for east and south orientation
    # but it's wrong for north and west orientations. But using the
    # default texture pack the torches are small enough to no overlap.
    alpha_over(img, torch, static_torch, torch)
    alpha_over(img, torch, moving_torch, torch)

    return img

# redstone comparator (149 is inactive, 150 is active)
@material(blockid=[149,150], data=list(range(16)), transparent=True, nospawn=True)
def comparator(self, blockid, data):

    # rotation
    # add self.rotation to the lower 2 bits,  mod 4
    data = data & 0b1100 | (((data & 0b11) + self.rotation) % 4)


    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/comparator") )) if blockid == 149 else self.process_texture(self.assetLoader.load_image(("minecraft:block/comparator_on") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/smooth_stone_slab_side") ))
    increment = 13

    if (data & 0x3) == 0: # pointing north
        pass
        static_torch = (-3,-1)
        torch = ((0,2),(6,-1))

    if (data & 0x3) == 1: # pointing east
        top = top.rotate(270)
        static_torch = (5,-1)
        torch = ((-4,-1),(0,2))

    if (data & 0x3) == 2: # pointing south
        top = top.rotate(180)
        static_torch = (5,3)
        torch = ((0,-4),(-4,-1))

    if (data & 0x3) == 3: # pointing west
        top = top.rotate(90)
        static_torch = (-3,3)
        torch = ((1,-4),(6,-1))


    def build_torch(active):
        # compose a "3d" redstone torch
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_torch_off") )).copy() if not active else self.process_texture(self.assetLoader.load_image(("minecraft:block/redstone_torch") )).copy()
        torch = Image.new("RGBA", (24,24), self.bgcolor)

        t_crop = t.crop((2,2,14,14))
        slice = t_crop.copy()
        ImageDraw.Draw(slice).rectangle((6,0,12,12),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(slice).rectangle((0,0,4,12),outline=(0,0,0,0),fill=(0,0,0,0))

        alpha_over(torch, slice, (6,4))
        alpha_over(torch, t_crop, (5,5))
        alpha_over(torch, t_crop, (6,5))
        alpha_over(torch, slice, (6,6))

        return torch

    active_torch = build_torch(True)
    inactive_torch = build_torch(False)
    back_torch = active_torch if (blockid == 150 or data & 0b1000 == 0b1000) else inactive_torch
    static_torch_img = active_torch if (data & 0b100 == 0b100) else inactive_torch

    img = self.build_full_block( (top, increment), None, None, side, side)

    alpha_over(img, static_torch_img, static_torch, static_torch_img)
    alpha_over(img, back_torch, torch[0], back_torch)
    alpha_over(img, back_torch, torch[1], back_torch)
    return img


# trapdoor
# the trapdoor is looks like a sprite when opened, that's not good
@material(blockid=[96,167,11332,11333,11334,11335,11336], data=list(range(16)), transparent=True, nospawn=True)
def trapdoor(self, blockid, data):

    # rotation
    # Masked to not clobber opened/closed info
    if self.rotation == 1:
        if (data & 0b0011) == 0: data = data & 0b1100 | 3
        elif (data & 0b0011) == 1: data = data & 0b1100 | 2
        elif (data & 0b0011) == 2: data = data & 0b1100 | 0
        elif (data & 0b0011) == 3: data = data & 0b1100 | 1
    elif self.rotation == 2:
        if (data & 0b0011) == 0: data = data & 0b1100 | 1
        elif (data & 0b0011) == 1: data = data & 0b1100 | 0
        elif (data & 0b0011) == 2: data = data & 0b1100 | 3
        elif (data & 0b0011) == 3: data = data & 0b1100 | 2
    elif self.rotation == 3:
        if (data & 0b0011) == 0: data = data & 0b1100 | 2
        elif (data & 0b0011) == 1: data = data & 0b1100 | 3
        elif (data & 0b0011) == 2: data = data & 0b1100 | 1
        elif (data & 0b0011) == 3: data = data & 0b1100 | 0

    # texture generation
    texturepath = {96:"minecraft:block/oak_trapdoor",
                   167:"minecraft:block/iron_trapdoor",
                   11332:"minecraft:block/spruce_trapdoor",
                   11333:"minecraft:block/birch_trapdoor",
                   11334:"minecraft:block/jungle_trapdoor",
                   11335:"minecraft:block/acacia_trapdoor",
                   11336:"minecraft:block/dark_oak_trapdoor"
                  }[blockid]

    if data & 0x4 == 0x4: # opened trapdoor
        if data & 0x08 == 0x08: texture = self.process_texture(self.assetLoader.load_image((texturepath) )).transpose(Image.FLIP_TOP_BOTTOM)
        else: texture = self.process_texture(self.assetLoader.load_image((texturepath) ))

        if data & 0x3 == 0: # west
            img = self.build_full_block(None, None, None, None, texture)
        if data & 0x3 == 1: # east
            img = self.build_full_block(None, texture, None, None, None)
        if data & 0x3 == 2: # south
            img = self.build_full_block(None, None, texture, None, None)
        if data & 0x3 == 3: # north
            img = self.build_full_block(None, None, None, texture, None)

    elif data & 0x4 == 0: # closed trapdoor
        texture = self.process_texture(self.assetLoader.load_image((texturepath) ))
        if data & 0x8 == 0x8: # is a top trapdoor
            img = Image.new("RGBA", (24,24), self.bgcolor)
            t = self.build_full_block((texture, 12), None, None, texture, texture)
            alpha_over(img, t, (0,-9),t)
        else: # is a bottom trapdoor
            img = self.build_full_block((texture, 12), None, None, texture, texture)

    return img

# block with hidden silverfish (stone, cobblestone and stone brick)
@material(blockid=97, data=list(range(3)), solid=True)
def hidden_silverfish(self, blockid, data):
    if data == 0: # stone
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone") ))
    elif data == 1: # cobblestone
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/cobblestone") ))
    elif data == 2: # stone brick
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone_bricks") ))

    img = self.build_block(t, t)

    return img

# stone brick
@material(blockid=98, data=list(range(4)), solid=True)
def stone_brick(self, blockid, data):
    if data == 0: # normal
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/stone_bricks") ))
    elif data == 1: # mossy
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/mossy_stone_bricks") ))
    elif data == 2: # cracked
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/cracked_stone_bricks") ))
    elif data == 3: # "circle" stone brick
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/chiseled_stone_bricks") ))

    img = self.build_full_block(t, None, None, t, t)

    return img

# huge brown and red mushroom
@material(blockid=[99,100], data= list(range(11)) + [14,15], solid=True)
def huge_mushroom(self, blockid, data):
    # rotation
    if self.rotation == 1:
        if data == 1: data = 3
        elif data == 2: data = 6
        elif data == 3: data = 9
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 1
        elif data == 8: data = 4
        elif data == 9: data = 7
    elif self.rotation == 2:
        if data == 1: data = 9
        elif data == 2: data = 8
        elif data == 3: data = 7
        elif data == 4: data = 6
        elif data == 6: data = 4
        elif data == 7: data = 3
        elif data == 8: data = 2
        elif data == 9: data = 1
    elif self.rotation == 3:
        if data == 1: data = 7
        elif data == 2: data = 4
        elif data == 3: data = 1
        elif data == 4: data = 2
        elif data == 6: data = 8
        elif data == 7: data = 9
        elif data == 8: data = 6
        elif data == 9: data = 3

    # texture generation
    if blockid == 99: # brown
        cap = self.process_texture(self.assetLoader.load_image(("minecraft:block/brown_mushroom_block") ))
    else: # red
        cap = self.process_texture(self.assetLoader.load_image(("minecraft:block/red_mushroom_block") ))

    stem = self.process_texture(self.assetLoader.load_image(("minecraft:block/mushroom_stem") ))
    porous = self.process_texture(self.assetLoader.load_image(("minecraft:block/mushroom_block_inside") ))

    if data == 0: # fleshy piece
        img = self.build_full_block(porous, None, None, porous, porous)

    if data == 1: # north-east corner
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 2: # east side
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 3: # south-east corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 4: # north side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 5: # top piece
        img = self.build_full_block(cap, None, None, porous, porous)

    if data == 6: # south side
        img = self.build_full_block(cap, None, None, cap, porous)

    if data == 7: # north-west corner
        img = self.build_full_block(cap, None, None, cap, cap)

    if data == 8: # west side
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 9: # south-west corner
        img = self.build_full_block(cap, None, None, porous, cap)

    if data == 10: # stem
        img = self.build_full_block(porous, None, None, stem, stem)

    if data == 14: # all cap
        img = self.build_block(cap,cap)

    if data == 15: # all stem
        img = self.build_block(stem,stem)

    return img

# iron bars and glass pane
# TODO glass pane is not a sprite, it has a texture for the side,
# at the moment is not used
@material(blockid=[101,102, 160], data=list(range(256)), transparent=True, nospawn=True)
def panes(self, blockid, data):
    # no rotation, uses pseudo data
    if blockid == 101:
        # iron bars
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/iron_bars") ))
    elif blockid == 160:
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_stained_glass"
                                                              % color_map[data & 0xf])))
    else:
        # glass panes
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/glass") ))
    left = t.copy()
    right = t.copy()

    # generate the four small pieces of the glass pane
    ImageDraw.Draw(right).rectangle((0,0,7,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(left).rectangle((8,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    up_left = self.transform_image_side(left)
    up_right = self.transform_image_side(right).transpose(Image.FLIP_TOP_BOTTOM)
    dw_right = self.transform_image_side(right)
    dw_left = self.transform_image_side(left).transpose(Image.FLIP_TOP_BOTTOM)

    # Create img to compose the texture
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # +x axis points top right direction
    # +y axis points bottom right direction
    # First compose things in the back of the image,
    # then things in the front.

    # the lower 4 bits encode color, the upper 4 encode adjencies
    data = data >> 4

    if (data & 0b0001) == 1 or data == 0:
        alpha_over(img,up_left, (6,3),up_left)    # top left
    if (data & 0b1000) == 8 or data == 0:
        alpha_over(img,up_right, (6,3),up_right)  # top right
    if (data & 0b0010) == 2 or data == 0:
        alpha_over(img,dw_left, (6,3),dw_left)    # bottom left
    if (data & 0b0100) == 4 or data == 0:
        alpha_over(img,dw_right, (6,3),dw_right)  # bottom right

    return img

# melon
block(blockid=103, top_image="minecraft:block/melon_top", side_image="minecraft:block/melon_side", solid=True)

# pumpkin and melon stem
# TODO To render it as in game needs from pseudo data and ancil data:
# once fully grown the stem bends to the melon/pumpkin block,
# at the moment only render the growing stem
@material(blockid=[104,105], data=list(range(8)), transparent=True)
def stem(self, blockid, data):
    # the ancildata value indicates how much of the texture
    # is shown.

    # not fully grown stem or no pumpkin/melon touching it,
    # straight up stem
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/melon_stem") )).copy()
    img = Image.new("RGBA", (16,16), self.bgcolor)
    alpha_over(img, t, (0, int(16 - 16*((data + 1)/8.))), t)
    img = self.build_sprite(t)
    if data & 7 == 7:
        # fully grown stem gets brown color!
        # there is a conditional in rendermode-normal.c to not
        # tint the data value 7
        img = self.tint_texture(img, (211,169,116))
    return img


# vines
@material(blockid=106, data=list(range(16)), transparent=True)
def vines(self, blockid, data):
    # rotation
    # vines data is bit coded. decode it first.
    # NOTE: the directions used in this function are the new ones used
    # in minecraft 1.0.0, no the ones used by overviewer
    # (i.e. north is top-left by defalut)

    # rotate the data by bitwise shift
    shifts = 0
    if self.rotation == 1:
        shifts = 1
    elif self.rotation == 2:
        shifts = 2
    elif self.rotation == 3:
        shifts = 3

    for i in range(shifts):
        data = data * 2
        if data & 16:
            data = (data - 16) | 1

    # decode data and prepare textures
    raw_texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/vine") ))
    s = w = n = e = None

    if data & 1: # south
        s = raw_texture
    if data & 2: # west
        w = raw_texture
    if data & 4: # north
        n = raw_texture
    if data & 8: # east
        e = raw_texture

    # texture generation
    img = self.build_full_block(None, n, e, w, s)

    return img

# fence gates
@material(blockid=[107, 183, 184, 185, 186, 187], data=list(range(8)), transparent=True, nospawn=True)
def fence_gate(self, blockid, data):

    # rotation
    opened = False
    if data & 0x4:
        data = data & 0x3
        opened = True
    if self.rotation == 1:
        if data == 0: data = 1
        elif data == 1: data = 2
        elif data == 2: data = 3
        elif data == 3: data = 0
    elif self.rotation == 2:
        if data == 0: data = 2
        elif data == 1: data = 3
        elif data == 2: data = 0
        elif data == 3: data = 1
    elif self.rotation == 3:
        if data == 0: data = 3
        elif data == 1: data = 0
        elif data == 2: data = 1
        elif data == 3: data = 2
    if opened:
        data = data | 0x4

    # create the closed gate side
    if blockid == 107: # Oak
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_planks") )).copy()
    elif blockid == 183: # Spruce
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_planks") )).copy()
    elif blockid == 184: # Birch
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_planks") )).copy()
    elif blockid == 185: # Jungle
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_planks") )).copy()
    elif blockid == 186: # Dark Oak
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_planks") )).copy()
    elif blockid == 187: # Acacia
        gate_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_planks") )).copy()
    else:
        return None

    gate_side_draw = ImageDraw.Draw(gate_side)
    gate_side_draw.rectangle((7,0,15,0),outline=(0,0,0,0),fill=(0,0,0,0))
    gate_side_draw.rectangle((7,4,9,6),outline=(0,0,0,0),fill=(0,0,0,0))
    gate_side_draw.rectangle((7,10,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    gate_side_draw.rectangle((0,12,15,16),outline=(0,0,0,0),fill=(0,0,0,0))
    gate_side_draw.rectangle((0,0,4,15),outline=(0,0,0,0),fill=(0,0,0,0))
    gate_side_draw.rectangle((14,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # darken the sides slightly, as with the fences
    sidealpha = gate_side.split()[3]
    gate_side = ImageEnhance.Brightness(gate_side).enhance(0.9)
    gate_side.putalpha(sidealpha)

    # create the other sides
    mirror_gate_side = self.transform_image_side(gate_side.transpose(Image.FLIP_LEFT_RIGHT))
    gate_side = self.transform_image_side(gate_side)
    gate_other_side = gate_side.transpose(Image.FLIP_LEFT_RIGHT)
    mirror_gate_other_side = mirror_gate_side.transpose(Image.FLIP_LEFT_RIGHT)

    # Create img to compose the fence gate
    img = Image.new("RGBA", (24,24), self.bgcolor)

    if data & 0x4:
        # opened
        data = data & 0x3
        if data == 0:
            alpha_over(img, gate_side, (2,8), gate_side)
            alpha_over(img, gate_side, (13,3), gate_side)
        elif data == 1:
            alpha_over(img, gate_other_side, (-1,3), gate_other_side)
            alpha_over(img, gate_other_side, (10,8), gate_other_side)
        elif data == 2:
            alpha_over(img, mirror_gate_side, (-1,7), mirror_gate_side)
            alpha_over(img, mirror_gate_side, (10,2), mirror_gate_side)
        elif data == 3:
            alpha_over(img, mirror_gate_other_side, (2,1), mirror_gate_other_side)
            alpha_over(img, mirror_gate_other_side, (13,7), mirror_gate_other_side)
    else:
        # closed

        # positions for pasting the fence sides, as with fences
        pos_top_left = (2,3)
        pos_top_right = (10,3)
        pos_bottom_right = (10,7)
        pos_bottom_left = (2,7)

        if data == 0 or data == 2:
            alpha_over(img, gate_other_side, pos_top_right, gate_other_side)
            alpha_over(img, mirror_gate_other_side, pos_bottom_left, mirror_gate_other_side)
        elif data == 1 or data == 3:
            alpha_over(img, gate_side, pos_top_left, gate_side)
            alpha_over(img, mirror_gate_side, pos_bottom_right, mirror_gate_side)

    return img

# mycelium
block(blockid=110, top_image="minecraft:block/mycelium_top", side_image="minecraft:block/mycelium_side")

# lilypad
# At the moment of writing this lilypads has no ancil data and their
# orientation depends on their position on the map. So it uses pseudo
# ancildata.
@material(blockid=111, data=list(range(4)), transparent=True)
def lilypad(self, blockid, data):
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/lily_pad") )).copy()
    if data == 0:
        t = t.rotate(180)
    elif data == 1:
        t = t.rotate(270)
    elif data == 2:
        t = t
    elif data == 3:
        t = t.rotate(90)

    return self.build_full_block(None, None, None, None, None, t)

# nether brick
block(blockid=112, top_image="minecraft:block/nether_bricks")

# nether wart
@material(blockid=115, data=list(range(4)), transparent=True)
def nether_wart(self, blockid, data):
    if data == 0: # just come up
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_wart_stage0") ))
    elif data in (1, 2):
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_wart_stage1") ))
    else: # fully grown
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/nether_wart_stage2") ))

    # use the same technic as tall grass
    img = self.build_billboard(t)

    return img

# enchantment table
# TODO there's no book at the moment
@material(blockid=116, transparent=True, nodata=True)
def enchantment_table(self, blockid, data):
    # no book at the moment
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/enchanting_table_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/enchanting_table_side") ))
    img = self.build_full_block((top, 4), None, None, side, side)

    return img

# brewing stand
# TODO this is a place holder, is a 2d image pasted
@material(blockid=117, data=list(range(5)), transparent=True)
def brewing_stand(self, blockid, data):
    base = self.process_texture(self.assetLoader.load_image(("minecraft:block/brewing_stand_base") ))
    img = self.build_full_block(None, None, None, None, None, base)
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/brewing_stand") ))
    stand = self.build_billboard(t)
    alpha_over(img,stand,(0,-2))
    return img

# cauldron
@material(blockid=118, data=list(range(4)), transparent=True)
def cauldron(self, blockid, data):
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/cauldron_side") ))
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/cauldron_top") ))
    bottom = self.process_texture(self.assetLoader.load_image(("minecraft:block/cauldron_inner") ))
    water = self.transform_image_top(self.load_water())
    if data == 0: # empty
        img = self.build_full_block(top, side, side, side, side)
    if data == 1: # 1/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,8), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)
    if data == 2: # 2/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,4), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)
    if data == 3: # 3/3 filled
        img = self.build_full_block(None , side, side, None, None)
        alpha_over(img, water, (0,0), water)
        img2 = self.build_full_block(top , None, None, side, side)
        alpha_over(img, img2, (0,0), img2)

    return img

# end portal and end_gateway
@material(blockid=[119,209], transparent=True, nodata=True)
def end_portal(self, blockid, data):
    img = Image.new("RGBA", (24,24), self.bgcolor)
    # generate a black texure with white, blue and grey dots resembling stars
    t = Image.new("RGBA", (16,16), (0,0,0,255))
    for color in [(155,155,155,255), (100,255,100,255), (255,255,255,255)]:
        for i in range(6):
            x = randint(0,15)
            y = randint(0,15)
            t.putpixel((x,y),color)
    if blockid == 209: # end_gateway
        return  self.build_block(t, t)

    t = self.transform_image_top(t)
    alpha_over(img, t, (0,0), t)

    return img

# end portal frame (data range 8 to get all orientations of filled)
@material(blockid=120, data=list(range(8)), transparent=True)
def end_portal_frame(self, blockid, data):
    # The bottom 2 bits are oritation info but seems there is no
    # graphical difference between orientations
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/end_portal_frame_top") ))
    eye_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/end_portal_frame_eye") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/end_portal_frame_side") ))
    img = self.build_full_block((top, 4), None, None, side, side)
    if data & 0x4 == 0x4: # ender eye on it
        # generate the eye
        eye_t = self.process_texture(self.assetLoader.load_image(("minecraft:block/end_portal_frame_eye") )).copy()
        eye_t_s = self.process_texture(self.assetLoader.load_image(("minecraft:block/end_portal_frame_eye") )).copy()
        # cut out from the texture the side and the top of the eye
        ImageDraw.Draw(eye_t).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
        ImageDraw.Draw(eye_t_s).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
        # trnasform images and paste
        eye = self.transform_image_top(eye_t)
        eye_s = self.transform_image_side(eye_t_s)
        eye_os = eye_s.transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, eye_s, (5,5), eye_s)
        alpha_over(img, eye_os, (9,5), eye_os)
        alpha_over(img, eye, (0,0), eye)

    return img

# end stone
block(blockid=121, top_image="minecraft:block/end_stone")

# dragon egg
# NOTE: this isn't a block, but I think it's better than nothing
block(blockid=122, top_image="minecraft:block/dragon_egg")

# inactive redstone lamp
block(blockid=123, top_image="minecraft:block/redstone_lamp")

# active redstone lamp
block(blockid=124, top_image="minecraft:block/redstone_lamp_on")

# daylight sensor.
@material(blockid=[151,178], transparent=True)
def daylight_sensor(self, blockid, data):
    if blockid == 151: # daylight sensor
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/daylight_detector_top") ))
    else: # inverted daylight sensor
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/daylight_detector_inverted_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/daylight_detector_side") ))

    # cut the side texture in half
    mask = side.crop((0,8,16,16))
    side = Image.new(side.mode, side.size, self.bgcolor)
    alpha_over(side, mask,(0,0,16,8), mask)

    # plain slab
    top = self.transform_image_top(top)
    side = self.transform_image_side(side)
    otherside = side.transpose(Image.FLIP_LEFT_RIGHT)

    sidealpha = side.split()[3]
    side = ImageEnhance.Brightness(side).enhance(0.9)
    side.putalpha(sidealpha)
    othersidealpha = otherside.split()[3]
    otherside = ImageEnhance.Brightness(otherside).enhance(0.8)
    otherside.putalpha(othersidealpha)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, side, (0,12), side)
    alpha_over(img, otherside, (12,12), otherside)
    alpha_over(img, top, (0,6), top)

    return img


# wooden double and normal slabs
# these are the new wooden slabs, blockids 43 44 still have wooden
# slabs, but those are unobtainable without cheating
@material(blockid=[125, 126], data=list(range(16)), transparent=(44,), solid=True)
def wooden_slabs(self, blockid, data):
    texture = data & 7
    if texture== 0: # oak
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/oak_planks") ))
    elif texture== 1: # spruce
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/spruce_planks") ))
    elif texture== 2: # birch
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/birch_planks") ))
    elif texture== 3: # jungle
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/jungle_planks") ))
    elif texture== 4: # acacia
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/acacia_planks") ))
    elif texture== 5: # dark wood
        top = side = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_oak_planks") ))
    else:
        return None

    if blockid == 125: # double slab
        return self.build_block(top, side)

    return self.build_slab_block(top, side, data & 8 == 8);

# emerald ore
block(blockid=129, top_image="minecraft:block/emerald_ore")

# emerald block
block(blockid=133, top_image="minecraft:block/emerald_block")

# cocoa plant
@material(blockid=127, data=list(range(12)), transparent=True)
def cocoa_plant(self, blockid, data):
    orientation = data & 3
    # rotation
    if self.rotation == 1:
        if orientation == 0: orientation = 1
        elif orientation == 1: orientation = 2
        elif orientation == 2: orientation = 3
        elif orientation == 3: orientation = 0
    elif self.rotation == 2:
        if orientation == 0: orientation = 2
        elif orientation == 1: orientation = 3
        elif orientation == 2: orientation = 0
        elif orientation == 3: orientation = 1
    elif self.rotation == 3:
        if orientation == 0: orientation = 3
        elif orientation == 1: orientation = 0
        elif orientation == 2: orientation = 1
        elif orientation == 3: orientation = 2

    size = data & 12
    if size == 8: # big
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/cocoa_stage2") ))
        c_left = (0,3)
        c_right = (8,3)
        c_top = (5,2)
    elif size == 4: # normal
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/cocoa_stage1") ))
        c_left = (-2,2)
        c_right = (8,2)
        c_top = (5,2)
    elif size == 0: # small
        t = self.process_texture(self.assetLoader.load_image(("minecraft:block/cocoa_stage0") ))
        c_left = (-3,2)
        c_right = (6,2)
        c_top = (5,2)

    # let's get every texture piece necessary to do this
    stalk = t.copy()
    ImageDraw.Draw(stalk).rectangle((0,0,11,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(stalk).rectangle((12,4,16,16),outline=(0,0,0,0),fill=(0,0,0,0))

    top = t.copy() # warning! changes with plant size
    ImageDraw.Draw(top).rectangle((0,7,16,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(top).rectangle((7,0,16,6),outline=(0,0,0,0),fill=(0,0,0,0))

    side = t.copy() # warning! changes with plant size
    ImageDraw.Draw(side).rectangle((0,0,6,16),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(side).rectangle((0,0,16,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(side).rectangle((0,14,16,16),outline=(0,0,0,0),fill=(0,0,0,0))

    # first compose the block of the cocoa plant
    block = Image.new("RGBA", (24,24), self.bgcolor)
    tmp = self.transform_image_side(side).transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over (block, tmp, c_right,tmp) # right side
    tmp = tmp.transpose(Image.FLIP_LEFT_RIGHT)
    alpha_over (block, tmp, c_left,tmp) # left side
    tmp = self.transform_image_top(top)
    alpha_over(block, tmp, c_top,tmp)
    if size == 0:
        # fix a pixel hole
        block.putpixel((6,9), block.getpixel((6,10)))

    # compose the cocoa plant
    img = Image.new("RGBA", (24,24), self.bgcolor)
    if orientation in (2,3): # south and west
        tmp = self.transform_image_side(stalk).transpose(Image.FLIP_LEFT_RIGHT)
        alpha_over(img, block,(-1,-2), block)
        alpha_over(img, tmp, (4,-2), tmp)
        if orientation == 3:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation in (0,1): # north and east
        tmp = self.transform_image_side(stalk.transpose(Image.FLIP_LEFT_RIGHT))
        alpha_over(img, block,(-1,5), block)
        alpha_over(img, tmp, (2,12), tmp)
        if orientation == 0:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img

# command block
@material(blockid=[137,210,211], solid=True, nodata=True)
def command_block(self, blockid, data):
    if blockid == 210:
        front = self.process_texture(self.assetLoader.load_image(("minecraft:block/repeating_command_block_front") ))
        side = self.process_texture(self.assetLoader.load_image(("minecraft:block/repeating_command_block_side") ))
        back = self.process_texture(self.assetLoader.load_image(("minecraft:block/repeating_command_block_back") ))
    elif blockid == 211:
        front = self.process_texture(self.assetLoader.load_image(("minecraft:block/chain_command_block_front") ))
        side = self.process_texture(self.assetLoader.load_image(("minecraft:block/chain_command_block_side") ))
        back = self.process_texture(self.assetLoader.load_image(("minecraft:block/chain_command_block_back") ))
    else:
        front = self.process_texture(self.assetLoader.load_image(("minecraft:block/command_block_front") ))
        side = self.process_texture(self.assetLoader.load_image(("minecraft:block/command_block_side") ))
        back = self.process_texture(self.assetLoader.load_image(("minecraft:block/command_block_back") ))
    return self.build_full_block(side, side, back, front, side)

# beacon block
# at the moment of writing this, it seems the beacon block doens't use
# the data values
@material(blockid=138, transparent=True, nodata = True)
def beacon(self, blockid, data):
    # generate the three pieces of the block
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/glass") ))
    glass = self.build_block(t,t)
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/obsidian") ))
    obsidian = self.build_full_block((t,12),None, None, t, t)
    obsidian = obsidian.resize((20,20), Image.ANTIALIAS)
    t = self.process_texture(self.assetLoader.load_image(("minecraft:block/beacon") ))
    crystal = self.build_block(t,t)
    crystal = crystal.resize((16,16),Image.ANTIALIAS)

    # compose the block
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, obsidian, (2, 4), obsidian)
    alpha_over(img, crystal, (4,3), crystal)
    alpha_over(img, glass, (0,0), glass)

    return img

# cobblestone and mossy cobblestone walls, chorus plants, mossy stone brick walls
# one additional bit of data value added for mossy and cobblestone
@material(blockid=[199, *range(21000,21013+1)], data=list(range(32)), transparent=True, nospawn=True)
def cobblestone_wall(self, blockid, data):
    walls_id_to_tex = {
          199: "minecraft:block/chorus_plant", # chorus plants
        21000: "minecraft:block/andesite",
        21001: "minecraft:block/bricks",
        21002: "minecraft:block/cobblestone",
        21003: "minecraft:block/diorite",
        21004: "minecraft:block/end_stone_bricks",
        21005: "minecraft:block/granite",
        21006: "minecraft:block/mossy_cobblestone",
        21007: "minecraft:block/mossy_stone_bricks",
        21008: "minecraft:block/nether_bricks",
        21009: "minecraft:block/prismarine",
        21010: "minecraft:block/red_nether_bricks",
        21011: "minecraft:block/red_sandstone",
        21012: "minecraft:block/sandstone",
        21013: "minecraft:block/stone_bricks"
    }
    t = self.process_texture(self.assetLoader.load_image((walls_id_to_tex[blockid]))).copy()

    wall_pole_top = t.copy()
    wall_pole_side = t.copy()
    wall_side_top = t.copy()
    wall_side = t.copy()
    # _full is used for walls without pole
    wall_side_top_full = t.copy()
    wall_side_full = t.copy()

    # generate the textures of the wall
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,0,15,3),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_top).rectangle((0,12,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    ImageDraw.Draw(wall_pole_side).rectangle((0,0,3,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_pole_side).rectangle((12,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # Create the sides and the top of the pole
    wall_pole_side = self.transform_image_side(wall_pole_side)
    wall_pole_other_side = wall_pole_side.transpose(Image.FLIP_LEFT_RIGHT)
    wall_pole_top = self.transform_image_top(wall_pole_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_pole_side.split()[3]
    wall_pole_side = ImageEnhance.Brightness(wall_pole_side).enhance(0.8)
    wall_pole_side.putalpha(sidealpha)
    othersidealpha = wall_pole_other_side.split()[3]
    wall_pole_other_side = ImageEnhance.Brightness(wall_pole_other_side).enhance(0.7)
    wall_pole_other_side.putalpha(othersidealpha)

    # Compose the wall pole
    wall_pole = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(wall_pole,wall_pole_side, (3,4),wall_pole_side)
    alpha_over(wall_pole,wall_pole_other_side, (9,4),wall_pole_other_side)
    alpha_over(wall_pole,wall_pole_top, (0,0),wall_pole_top)

    # create the sides and the top of a wall attached to a pole
    ImageDraw.Draw(wall_side).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,11,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,0,15,4),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top).rectangle((0,11,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    # full version, without pole
    ImageDraw.Draw(wall_side_full).rectangle((0,0,15,2),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(wall_side_top_full).rectangle((0,4,15,15),outline=(0,0,0,0),fill=(0,0,0,0))

    # compose the sides of a wall atached to a pole
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side = self.transform_image_side(wall_side)
    wall_side_top = self.transform_image_top(wall_side_top)

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side.split()[3]
    wall_side = ImageEnhance.Brightness(wall_side).enhance(0.7)
    wall_side.putalpha(sidealpha)

    alpha_over(tmp,wall_side, (0,0),wall_side)
    alpha_over(tmp,wall_side_top, (-5,3),wall_side_top)
    wall_side = tmp
    wall_other_side = wall_side.transpose(Image.FLIP_LEFT_RIGHT)

    # compose the sides of the full wall
    tmp = Image.new("RGBA", (24,24), self.bgcolor)
    wall_side_full = self.transform_image_side(wall_side_full)
    wall_side_top_full = self.transform_image_top(wall_side_top_full.rotate(90))

    # Darken the sides slightly. These methods also affect the alpha layer,
    # so save them first (we don't want to "darken" the alpha layer making
    # the block transparent)
    sidealpha = wall_side_full.split()[3]
    wall_side_full = ImageEnhance.Brightness(wall_side_full).enhance(0.7)
    wall_side_full.putalpha(sidealpha)

    alpha_over(tmp,wall_side_full, (4,0),wall_side_full)
    alpha_over(tmp,wall_side_top_full, (3,-4),wall_side_top_full)
    wall_side_full = tmp
    wall_other_side_full = wall_side_full.transpose(Image.FLIP_LEFT_RIGHT)

    # Create img to compose the wall
    img = Image.new("RGBA", (24,24), self.bgcolor)

    # Position wall imgs around the wall bit stick
    pos_top_left = (-5,-2)
    pos_bottom_left = (-8,4)
    pos_top_right = (5,-3)
    pos_bottom_right = (7,4)

    # +x axis points top right direction
    # +y axis points bottom right direction
    # There are two special cases for wall without pole.
    # Normal case:
    # First compose the walls in the back of the image,
    # then the pole and then the walls in the front.
    if (data == 0b1010) or (data == 0b11010):
        alpha_over(img, wall_other_side_full,(0,2), wall_other_side_full)
    elif (data == 0b0101) or (data == 0b10101):
        alpha_over(img, wall_side_full,(0,2), wall_side_full)
    else:
        if (data & 0b0001) == 1:
            alpha_over(img,wall_side, pos_top_left,wall_side)                # top left
        if (data & 0b1000) == 8:
            alpha_over(img,wall_other_side, pos_top_right,wall_other_side)    # top right

        alpha_over(img,wall_pole,(0,0),wall_pole)

        if (data & 0b0010) == 2:
            alpha_over(img,wall_other_side, pos_bottom_left,wall_other_side)      # bottom left
        if (data & 0b0100) == 4:
            alpha_over(img,wall_side, pos_bottom_right,wall_side)                  # bottom right

    return img

# carrots, potatoes
@material(blockid=[141,142], data=list(range(8)), transparent=True, nospawn=True)
def crops4(self, blockid, data):
    # carrots and potatoes have 8 data, but only 4 visual stages
    stage = {0:0,
             1:0,
             2:1,
             3:1,
             4:2,
             5:2,
             6:2,
             7:3}[data]
    if blockid == 141: # carrots
        raw_crop = self.process_texture(self.assetLoader.load_image(("minecraft:block/carrots_stage%d" % stage) ))
    else: # potatoes
        raw_crop = self.process_texture(self.assetLoader.load_image(("minecraft:block/potatoes_stage%d" % stage) ))
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# anvils
@material(blockid=145, data=list(range(12)), transparent=True)
def anvil(self, blockid, data):

    # anvils only have two orientations, invert it for rotations 1 and 3
    orientation = data & 0x1
    if self.rotation in (1,3):
        if orientation == 1:
            orientation = 0
        else:
            orientation = 1

    # get the correct textures
    # the bits 0x4 and 0x8 determine how damaged is the anvil
    if (data & 0xc) == 0: # non damaged anvil
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/anvil_top") ))
    elif (data & 0xc) == 0x4: # slightly damaged
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/chipped_anvil_top") ))
    elif (data & 0xc) == 0x8: # very damaged
        top = self.process_texture(self.assetLoader.load_image(("minecraft:block/damaged_anvil_top") ))
    # everything else use this texture
    big_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/anvil") )).copy()
    small_side = self.process_texture(self.assetLoader.load_image(("minecraft:block/anvil") )).copy()
    base = self.process_texture(self.assetLoader.load_image(("minecraft:block/anvil") )).copy()
    small_base = self.process_texture(self.assetLoader.load_image(("minecraft:block/anvil") )).copy()

    # cut needed patterns
    ImageDraw.Draw(big_side).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((0,0,2,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((13,0,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(small_side).rectangle((0,8,15,15),outline=(0,0,0,0),fill=(0,0,0,0))
    ImageDraw.Draw(base).rectangle((0,0,15,15),outline=(0,0,0,0))
    ImageDraw.Draw(base).rectangle((1,1,14,14),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((0,0,15,15),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((1,1,14,14),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((2,2,13,13),outline=(0,0,0,0))
    ImageDraw.Draw(small_base).rectangle((3,3,12,12),outline=(0,0,0,0))

    # check orientation and compose the anvil
    if orientation == 1: # bottom-left top-right
        top = top.rotate(90)
        left_side = small_side
        left_pos = (1,7)
        right_side = big_side
        right_pos = (10,5)
    else: # top-left bottom-right
        right_side = small_side
        right_pos = (12,7)
        left_side = big_side
        left_pos = (3,5)

    img = Image.new("RGBA", (24,24), self.bgcolor)

    # darken sides
    alpha = big_side.split()[3]
    big_side = ImageEnhance.Brightness(big_side).enhance(0.8)
    big_side.putalpha(alpha)
    alpha = small_side.split()[3]
    small_side = ImageEnhance.Brightness(small_side).enhance(0.9)
    small_side.putalpha(alpha)
    alpha = base.split()[3]
    base_d = ImageEnhance.Brightness(base).enhance(0.8)
    base_d.putalpha(alpha)

    # compose
    base = self.transform_image_top(base)
    base_d = self.transform_image_top(base_d)
    small_base = self.transform_image_top(small_base)
    top = self.transform_image_top(top)

    alpha_over(img, base_d, (0,12), base_d)
    alpha_over(img, base_d, (0,11), base_d)
    alpha_over(img, base_d, (0,10), base_d)
    alpha_over(img, small_base, (0,10), small_base)

    alpha_over(img, top, (0,0), top)

    left_side = self.transform_image_side(left_side)
    right_side = self.transform_image_side(right_side).transpose(Image.FLIP_LEFT_RIGHT)

    alpha_over(img, left_side, left_pos, left_side)
    alpha_over(img, right_side, right_pos, right_side)

    return img


# block of redstone
block(blockid=152, top_image="minecraft:block/redstone_block")

# nether quartz ore
block(blockid=153, top_image="minecraft:block/nether_quartz_ore")

# block of quartz
@material(blockid=155, data=list(range(5)), solid=True)
def quartz_block(self, blockid, data):

    if data in (0,1): # normal and chiseled quartz block
        if data == 0:
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/quartz_block_top") ))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/quartz_block_side") ))
        else:
            top = self.process_texture(self.assetLoader.load_image(("minecraft:block/chiseled_quartz_block_top") ))
            side = self.process_texture(self.assetLoader.load_image(("minecraft:block/chiseled_quartz_block") ))
        return self.build_block(top, side)

    # pillar quartz block with orientation
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/quartz_pillar_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/quartz_pillar") )).copy()
    if data == 2: # vertical
        return self.build_block(top, side)
    elif data == 3: # north-south oriented
        if self.rotation in (0,2):
            return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
        return self.build_full_block(side, None, None, side.rotate(90), top)

    elif data == 4: # east-west oriented
        if self.rotation in (0,2):
            return self.build_full_block(side, None, None, side.rotate(90), top)
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))

# hopper
@material(blockid=154, data=list(range(4)), transparent=True)
def hopper(self, blockid, data):
    #build the top
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/hopper_outside") ))
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/hopper_top") ))
    bottom = self.process_texture(self.assetLoader.load_image(("minecraft:block/hopper_inside") ))
    hop_top = self.build_full_block((top,10), side, side, side, side, side)

    #build a solid block for mid/top
    hop_mid = self.build_full_block((top,5), side, side, side, side, side)
    hop_bot = self.build_block(side,side)

    hop_mid = hop_mid.resize((17,17),Image.ANTIALIAS)
    hop_bot = hop_bot.resize((10,10),Image.ANTIALIAS)

    #compose the final block
    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, hop_bot, (7,14), hop_bot)
    alpha_over(img, hop_mid, (3,3), hop_mid)
    alpha_over(img, hop_top, (0,-6), hop_top)

    return img

# slime block
block(blockid=165, top_image="minecraft:block/slime_block")

# prismarine block
@material(blockid=168, data=list(range(3)), solid=True)
def prismarine_block(self, blockid, data):

   if data == 0: # prismarine
       t = self.process_texture(self.assetLoader.load_image(("minecraft:block/prismarine") ))
   elif data == 1: # prismarine bricks
       t = self.process_texture(self.assetLoader.load_image(("minecraft:block/prismarine_bricks") ))
   elif data == 2: # dark prismarine
       t = self.process_texture(self.assetLoader.load_image(("minecraft:block/dark_prismarine") ))

   img = self.build_block(t, t)

   return img

# sea lantern
block(blockid=169, top_image="minecraft:block/sea_lantern")

# hay block
@material(blockid=170, data=list(range(9)), solid=True)
def hayblock(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/hay_block_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/hay_block_side") ))

    if self.rotation == 1:
        if data == 4: data = 8
        elif data == 8: data = 4
    elif self.rotation == 3:
        if data == 4: data = 8
        elif data == 8: data = 4

    # choose orientation and paste textures
    if data == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif data == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(90), top)
    else:
        return self.build_block(top, side)


# carpet - wool block that's small?
@material(blockid=171, data=list(range(16)), transparent=True)
def carpet(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_wool" %
                                                                color_map[data])))

    return self.build_full_block((texture,15),texture,texture,texture,texture)

#clay block
block(blockid=172, top_image="minecraft:block/terracotta")

#stained hardened clay
@material(blockid=159, data=list(range(16)), solid=True)
def stained_clay(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_terracotta" %
                                                                color_map[data])))

    return self.build_block(texture,texture)

#coal block
block(blockid=173, top_image="minecraft:block/coal_block")

# packed ice block
block(blockid=174, top_image="minecraft:block/packed_ice")

#blue ice
block(blockid=11312, top_image="minecraft:block/blue_ice")

#smooth stones
block(blockid=11313, top_image="minecraft:block/smooth_stone") # stone
block(blockid=11314, top_image="minecraft:block/sandstone_top") # sandstone
block(blockid=11315, top_image="minecraft:block/red_sandstone_top") # red sandstone

#coral block
block(blockid=11316, top_image="minecraft:block/brain_coral_block")
block(blockid=11317, top_image="minecraft:block/bubble_coral_block")
block(blockid=11318, top_image="minecraft:block/fire_coral_block")
block(blockid=11319, top_image="minecraft:block/horn_coral_block")
block(blockid=11320, top_image="minecraft:block/tube_coral_block")

#dead coral block
block(blockid=11321, top_image="minecraft:block/dead_brain_coral_block")
block(blockid=11322, top_image="minecraft:block/dead_bubble_coral_block")
block(blockid=11323, top_image="minecraft:block/dead_fire_coral_block")
block(blockid=11324, top_image="minecraft:block/dead_horn_coral_block")
block(blockid=11325, top_image="minecraft:block/dead_tube_coral_block")

@material(blockid=175, data=list(range(16)), transparent=True)
def flower(self, blockid, data):
    double_plant_map = ["sunflower", "lilac", "tall_grass", "large_fern", "rose_bush", "peony", "peony", "peony"]
    plant = double_plant_map[data & 0x7]

    if data & 0x8:
        part = "top"
    else:
        part = "bottom"

    png = "minecraft:block/%s_%s" % (plant,part)
    texture = self.process_texture(self.assetLoader.load_image((png)))
    img = self.build_billboard(texture)

    #sunflower top
    if data == 8:
        bloom_tex = self.process_texture(self.assetLoader.load_image(("minecraft:block/sunflower_front") ))
        alpha_over(img, bloom_tex.resize((14, 11), Image.ANTIALIAS), (5,5))

    return img

# chorus flower
@material(blockid=200, data=list(range(6)), solid=True)
def chorus_flower(self, blockid, data):
    # aged 5, dead
    if data == 5:
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/chorus_flower_dead") ))
    else:
        texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/chorus_flower") ))

    return self.build_block(texture,texture)

# purpur block
block(blockid=201, top_image="minecraft:block/purpur_block")

# purpur pilar
@material(blockid=202, data=list(range(12)) , solid=True)
def purpur_pillar(self, blockid, data):
    pillar_orientation = data & 12
    top=self.process_texture(self.assetLoader.load_image(("minecraft:block/purpur_pillar_top") ))
    side=self.process_texture(self.assetLoader.load_image(("minecraft:block/purpur_pillar") ))
    if pillar_orientation == 0: # east-west orientation
        return self.build_block(top, side)
    elif pillar_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif pillar_orientation == 8: # north-south orientation

        return self.build_full_block(side, None, None, side.rotate(270), top)

# end brick
block(blockid=206, top_image="minecraft:block/end_stone_bricks")

# frosted ice
@material(blockid=212, data=list(range(4)), solid=True)
def frosted_ice(self, blockid, data):
    img = self.process_texture(self.assetLoader.load_image(("minecraft:block/frosted_ice_%d" %
                                                            data)))
    return self.build_block(img, img)

# magma block
block(blockid=213, top_image="minecraft:block/magma")

# nether wart block
block(blockid=214, top_image="minecraft:block/nether_wart_block")

# red nether brick
block(blockid=215, top_image="minecraft:block/red_nether_bricks")

@material(blockid=216, data=list(range(12)), solid=True)
def boneblock(self, blockid, data):
    # extract orientation
    boneblock_orientation = data & 12
    if self.rotation == 1:
        if boneblock_orientation == 4: boneblock_orientation = 8
        elif boneblock_orientation == 8: boneblock_orientation = 4
    elif self.rotation == 3:
        if boneblock_orientation == 4: boneblock_orientation = 8
        elif boneblock_orientation == 8: boneblock_orientation = 4

    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/bone_block_top") ))
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/bone_block_side") ))

    # choose orientation and paste textures
    if boneblock_orientation == 0:
        return self.build_block(top, side)
    elif boneblock_orientation == 4: # east-west orientation
        return self.build_full_block(side.rotate(90), None, None, top, side.rotate(90))
    elif boneblock_orientation == 8: # north-south orientation
        return self.build_full_block(side, None, None, side.rotate(270), top)

# observer
@material(blockid=218, data=list(range(6)), solid=True, nospawn=True)
def observer(self, blockid, data):
    # first, do the rotation if needed
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    front = self.process_texture(self.assetLoader.load_image(("minecraft:block/observer_front") )).copy()
    side = self.process_texture(self.assetLoader.load_image(("minecraft:block/observer_side") )).copy()
    back = self.process_texture(self.assetLoader.load_image(("minecraft:block/observer_back") )).copy()
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/observer_top") )).copy()

    if data == 0: # down
        side = side.rotate(90)
        img = self.build_full_block(back, None, None, side, top)
    elif data == 1: # up
        side = side.rotate(90)
        img = self.build_full_block(front.rotate(180), None, None, side, top.rotate(180))
    elif data == 2: # east
        img = self.build_full_block(top.rotate(180), None, None, side, back)
    elif data == 3: # west
        img = self.build_full_block(top, None, None, side, front)
    elif data == 4: # north
        img = self.build_full_block(top.rotate(270), None, None, front, side)
    elif data == 5: # south
        img = self.build_full_block(top.rotate(90), None, None, back, side)

    return img

# shulker box
@material(blockid=list(range(219,235)), data=list(range(6)), solid=True, nospawn=True)
def shulker_box(self, blockid, data):
    # first, do the rotation if needed
    data = data & 7
    if self.rotation == 1:
        if data == 2: data = 5
        elif data == 3: data = 4
        elif data == 4: data = 2
        elif data == 5: data = 3
    elif self.rotation == 2:
        if data == 2: data = 3
        elif data == 3: data = 2
        elif data == 4: data = 5
        elif data == 5: data = 4
    elif self.rotation == 3:
        if data == 2: data = 4
        elif data == 3: data = 5
        elif data == 4: data = 3
        elif data == 5: data = 2

    color = color_map[blockid - 219]
    shulker_t = self.process_texture(self.assetLoader.load_image(("minecraft:entity/shulker/shulker_%s" % color) )).copy()
    w,h = shulker_t.size
    res = w // 4
    # Cut out the parts of the shulker texture we need for the box
    top = shulker_t.crop((res, 0, res * 2, res))
    bottom = shulker_t.crop((res * 2, int(res * 1.75), res * 3, int(res * 2.75)))
    side_top = shulker_t.crop((0, res, res, int(res * 1.75)))
    side_bottom = shulker_t.crop((0, int(res * 2.75), res, int(res * 3.25)))
    side = Image.new('RGBA', (res, res))
    side.paste(side_top, (0, 0), side_top)
    side.paste(side_bottom, (0, res // 2), side_bottom)

    if data == 0: # down
        side = side.rotate(180)
        img = self.build_full_block(bottom, None, None, side, side)
    elif data == 1: # up
        img = self.build_full_block(top, None, None, side, side)
    elif data == 2: # east
        img = self.build_full_block(side, None, None, side.rotate(90), bottom)
    elif data == 3: # west
        img = self.build_full_block(side.rotate(180), None, None, side.rotate(270), top)
    elif data == 4: # north
        img = self.build_full_block(side.rotate(90), None, None, top, side.rotate(270))
    elif data == 5: # south
        img = self.build_full_block(side.rotate(270), None, None, bottom, side.rotate(90))

    return img

# structure block
@material(blockid=255, data=list(range(4)), solid=True)
def structure_block(self, blockid, data):
    if data == 0:
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/structure_block_save") ))
    elif data == 1:
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/structure_block_load") ))
    elif data == 2:
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/structure_block_corner") ))
    elif data == 3:
        img = self.process_texture(self.assetLoader.load_image(("minecraft:block/structure_block_data") ))
    return self.build_block(img, img)

# beetroots
@material(blockid=207, data=list(range(4)), transparent=True, nospawn=True)
def crops(self, blockid, data):
    raw_crop = self.process_texture(self.assetLoader.load_image(("minecraft:block/beetroots_stage%d" % data) ))
    crop1 = self.transform_image_top(raw_crop)
    crop2 = self.transform_image_side(raw_crop)
    crop3 = crop2.transpose(Image.FLIP_LEFT_RIGHT)

    img = Image.new("RGBA", (24,24), self.bgcolor)
    alpha_over(img, crop1, (0,12), crop1)
    alpha_over(img, crop2, (6,3), crop2)
    alpha_over(img, crop3, (6,3), crop3)
    return img

# Concrete
@material(blockid=251, data=list(range(16)), solid=True)
def concrete(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_concrete" % color_map[data]) ))
    return self.build_block(texture, texture)

# Concrete Powder
@material(blockid=252, data=list(range(16)), solid=True)
def concrete(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_concrete_powder" % color_map[data]) ))
    return self.build_block(texture, texture)

# Glazed Terracotta
@material(blockid=list(range(235,251)), data=list(range(8)), solid=True)
def glazed_terracotta(self, blockid, data):
    texture = self.process_texture(self.assetLoader.load_image(("minecraft:block/%s_glazed_terracotta" % color_map[blockid - 235]) ))
    glazed_terracotta_orientation = data & 3

    # Glazed Terracotta rotations are need seperate handling for each render direction

    if self.rotation == 0: # rendering north upper-left
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))

    elif self.rotation == 1: # north upper-right
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_block(texture, texture)


    elif self.rotation == 2: # north lower-right
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))

    elif self.rotation == 3: # north lower-left
        # choose orientation and paste textures
        if glazed_terracotta_orientation == 0: # south / Player was facing North
            return self.build_full_block(texture.rotate(90), None, None, texture.rotate(270), texture.rotate(90))
        elif glazed_terracotta_orientation == 1: # west / Player was facing east
            return self.build_block(texture, texture)
        elif glazed_terracotta_orientation == 2: # North / Player was facing South
            return self.build_full_block(texture.rotate(270), None, None, texture.rotate(90), texture.rotate(270))
        elif glazed_terracotta_orientation == 3: # east / Player was facing west
            return self.build_full_block(texture.rotate(180), None, None, texture.rotate(180), texture.rotate(180))

# dried kelp block
@material(blockid=11331, data=[0], solid=True)
def sandstone(self, blockid, data):
    top = self.process_texture(self.assetLoader.load_image(("minecraft:block/dried_kelp_top") ))
    return self.build_block(top, self.process_texture(self.assetLoader.load_image((
        "minecraft:block/dried_kelp_side"))))

# scaffolding
block(blockid=11414, top_image="minecraft:block/scaffolding_top", side_image="minecraft:block/scaffolding_side", solid=False, transparent=True)

# beehive and bee_nest
@material(blockid=[11501, 11502], data=list(range(8)), solid=True)
def beehivenest(self, blockid, data):    
    if blockid == 11501: #beehive
        t_top = self.assetLoader.load_image("minecraft:block/beehive_end")
        t_side = self.assetLoader.load_image("minecraft:block/beehive_side")
        t_front = self.assetLoader.load_image("minecraft:block/beehive_front")
        t_front_honey = self.assetLoader.load_image("minecraft:block/beehive_front_honey")
    elif blockid == 11502: #bee_nest
        t_top = self.assetLoader.load_image("minecraft:block/bee_nest_top")
        t_side = self.assetLoader.load_image("minecraft:block/bee_nest_side")
        t_front = self.assetLoader.load_image("minecraft:block/bee_nest_front")
        t_front_honey = self.assetLoader.load_image("minecraft:block/bee_nest_front_honey")

    if data >= 4:
        front = t_front_honey
    else:
        front = t_front

    if self.rotation == 0: # rendering north upper-left
        if data == 0 or data == 4: # south
            return self.build_full_block(t_top, t_side, t_side, t_side, front)
        elif data == 1 or data == 5: # west
            return self.build_full_block(t_top, t_side, t_side, front, t_side)
        elif data == 2 or data == 6: # north
            return self.build_full_block(t_top, t_side, front, t_side, t_side)
        elif data == 3 or data == 7: # east
            return self.build_full_block(t_top, front, t_side, t_side, t_side)

    elif self.rotation == 1: # north upper-right
        if data == 0 or data == 4: # south
            return self.build_full_block(t_top, t_side, t_side, front, t_side)
        elif data == 1 or data == 5: # west
            return self.build_full_block(t_top, t_side, front, t_side, t_side)
        elif data == 2 or data == 6: # north
            return self.build_full_block(t_top, front, t_side, t_side, t_side)
        elif data == 3 or data == 7: # east
            return self.build_full_block(t_top, t_side, t_side, t_side, front)            

    elif self.rotation == 2: # north lower-right
        if data == 0 or data == 4: # south
            return self.build_full_block(t_top, t_side, front, t_side, t_side)
        elif data == 1 or data == 5: # west
            return self.build_full_block(t_top, front, t_side, t_side, t_side)
        elif data == 2 or data == 6: # north
            return self.build_full_block(t_top, t_side, t_side, t_side, front)
        elif data == 3 or data == 7: # east
            return self.build_full_block(t_top, t_side, t_side, front, t_side)
            
    elif self.rotation == 3: # north lower-left
        if data == 0 or data == 4: # south
            return self.build_full_block(t_top, front, t_side, t_side, t_side)
        elif data == 1 or data == 5: # west
            return self.build_full_block(t_top, t_side, t_side, t_side, front)
        elif data == 2 or data == 6: # north
            return self.build_full_block(t_top, t_side, t_side, front, t_side)
        elif data == 3 or data == 7: # east
            return self.build_full_block(t_top, t_side, front, t_side, t_side)

# honeycomb_block
block(blockid=11503, top_image="minecraft:block/honeycomb_block")

# honey_block
block(blockid=11504, top_image="minecraft:block/honey_block_top", side_image="minecraft:block/honey_block_side")
