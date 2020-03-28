import imp
import json
import logging
import os
import re
import sys
import zipfile
from collections import OrderedDict
from io import BytesIO

import PIL.Image as Image
import typing

from overviewer_core import util

from functools import lru_cache
logger = logging.getLogger(__name__)

class AssetLoaderException(Exception):
    "To be thrown when a texture is not found."
    pass


class AssetLoader(object):
    """Object manages all tasks related to loading files and config data from texturepacks and jar files
    """
    BLOCKSTATES_DIR = "assets/minecraft/blockstates"
    MODELS_DIR = "assets/minecraft/models"
    TEXTURES_DIR = "assets/minecraft/textures"
    def __init__(self, texturepath):
        self.texturepath = texturepath
        self.jars = OrderedDict()

    def load_file(self, path: str, name: str, ext: str = ""):
        logger.debug("Path: {0}".format(path))
        logger.debug("Name: {0} ".format(name))
        logger.debug("Ext: {0} ".format(ext))
        if ":" in name:
            name = name.split(":")[1]
        if not os.path.splitext(name)[1]: #doesnt have extension
            name = "{0}{1}".format(name, ext)
        if  name.count("/") >1: #name probably already contains a fully qualified path
            fp = "{1}".format(path, name)
        else:
            fp = "{0}/{1}".format(path, name)

        print("Complete path: {0}".format(fp))
        return self.find_file(fp, verbose=True)


    def walk_assets(self, path: str, filter: r"", ignore_unsupported_blocks=True):
        """Walk Assets directory in order of precedence in order to find all blocks"""
        #todo: test
        _ret = set()
        print(path)
        if self.texturepath:
            if (self.texturepath not in self.jars
                    and os.path.isfile(self.texturepath)):
                # Must be a resource pack. Look for the requested file within
                # it.

                pack = zipfile.ZipFile(self.texturepath)
                # pack.getinfo() will raise KeyError if the file is
                # not found.
                # pprint(pack.infolist())
                for i in pack.infolist():
                    if path in i.filename:

                        # pack.getinfo(path)
                        # logging.debug("Found %s in '%s'", path,
                        #                          self.textures.find_file_local_path)
                        self.jars[self.texturepath] = pack
                        # ok cool now move this to the start so we pick it first
                        self.jars.move_to_end(self.texturepath, last=False)
                    # return pack.open(path)

            elif os.path.isdir(self.texturepath):
                full_path = os.path.join(self.texturepath, path)

                logger.debug(path)
                for root, dir, files in os.walk(self.texturepath + "/" + path):
                    logger.debug(files)
                    for fn in files:
                        # logger.debug(os.path.join(root,files))
                        # logger.debug(filter)
                        # logger.debug(fn)
                        if re.search(filter,str(fn)):
                            _ret.add(os.path.splitext(fn)[0])

                logger.debug(_ret)
                return _ret
                    # return open(full_path, mode)

        if len(self.jars) > 0:
            # optimize: most likely can be sped up significantly
            for jarpath in self.jars:
                try:
                    jar = self.jars[jarpath]
                    infolist = jar.infolist()
                    for i in infolist:
                        # logging.info(i)
                        if bool(re.search(filter, i.filename)) & (path in i.filename):
                            _ret.add(os.path.splitext(os.path.split(i.filename)[1])[0])
                    logging.debug("Found (cached) %s in '%s'", path,
                                             jarpath)
                    # return jar.open(filename)
                except (KeyError, IOError) as e:
                    pass
        # pprint(_ret)
        return _ret

    # @lru_cache
    # def load_image(self, filename)->Image:
    #     """Returns an image object"""
    #
    #     # try:
    #     #     img = self.texture_cache[filename]
    #     #     if isinstance(img, Exception):  # Did we cache an exception?
    #     #         raise img                   # Okay then, raise it.
    #     #     return img
    #     # except KeyError:
    #     #     pass
    #
    #     try:
    #         fileobj = self.find_file(filename)
    #     except (AssetLoaderException, IOError) as e:
    #         # We cache when our good friend find_file can't find
    #         # a texture, so that we do not repeatedly search for it.
    #         # self.texture_cache[filename] = e
    #         raise e
    #     buffer = BytesIO(fileobj.read())
    #     img = Image.open(buffer).convert("RGBA")
    #     # self.texture_cache[filename] = img
    #     return img

    @lru_cache()
    def load_img(self, texture_name, ext=".png")->Image:

        with self.load_file( self.TEXTURES_DIR, texture_name, ext) as f:
            buffer =  Image.open(f)
            # buffer = BytesIO(fileobj.read())
            h, w =buffer.size #get images size
            if h != w:# check image is square if not (for example due to animated texture) crop shorter side
                buffer = buffer.crop((0,0,min(h,w),min(h,w)))
            texture = buffer.convert("RGBA")
            return texture

    def find_file(self, filename, mode="rb", verbose=False)->typing.IO:
        """Searches for the given file and returns an open handle to it.
        This searches the following locations in this order:

        * In the directory textures_path given in the initializer if not already open
        * In an already open resource pack or client jar file
        * In the resource pack given by textures_path
        * The program dir (same dir as overviewer.py) for extracted textures
        * On Darwin, in /Applications/Minecraft for extracted textures
        * Inside a minecraft client jar. Client jars are searched for in the
          following location depending on platform:

            * On Windows, at %APPDATA%/.minecraft/versions/
            * On Darwin, at
                $HOME/Library/Application Support/minecraft/versions
            * at $HOME/.minecraft/versions/

          Only the latest non-snapshot version >1.6 is used

        * The overviewer_core/data/textures dir

        """
        if verbose: logging.info("Starting search for {0}".format(filename))

        # A texture path was given on the command line. Search this location
        # for the file first.
        if self.texturepath:
            if (self.texturepath not in self.jars
                    and os.path.isfile(self.texturepath)):
                # Must be a resource pack. Look for the requested file within
                # it.
                try:
                    pack = zipfile.ZipFile(self.texturepath)
                    # pack.getinfo() will raise KeyError if the file is
                    # not found.
                    pack.getinfo(filename)
                    if verbose: logging.info("Found %s in '%s'", filename,
                                             self.texturepath)
                    self.jars[self.texturepath] = pack
                    # ok cool now move this to the start so we pick it first
                    self.jars.move_to_end(self.texturepath, last=False)
                    return pack.open(filename)
                except (zipfile.BadZipfile, KeyError, IOError):
                    pass
            elif os.path.isdir(self.texturepath):
                full_path = os.path.join(self.texturepath, filename)
                if os.path.isfile(full_path):
                    if verbose: logging.info("Found %s in '%s'", filename, full_path)
                    return open(full_path, mode)

        # We already have some jars open, better use them.
        if len(self.jars) > 0:
            for jarpath in self.jars:
                try:
                    jar = self.jars[jarpath]
                    jar.getinfo(filename)
                    if verbose: logging.info("Found (cached) %s in '%s'", filename,
                                             jarpath)
                    return jar.open(filename)
                except (KeyError, IOError) as e:
                    pass

        # If we haven't returned at this point, then the requested file was NOT
        # found in the user-specified texture path or resource pack.
        if verbose: logging.info("Did not find the file in specified texture path")


        # Look in the location of the overviewer executable for the given path
        programdir = util.get_program_path()
        path = os.path.join(programdir,"data", filename)
        if os.path.isfile(path):
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)

        if sys.platform.startswith("darwin"):
            path = os.path.join("/Applications/Minecraft", filename)
            if os.path.isfile(path):
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)

        if verbose: logging.info("Did not find the file in overviewer executable directory")
        if verbose: logging.info("Looking for installed minecraft jar files...")

        # Find an installed minecraft client jar and look in it for the texture
        # file we need.
        versiondir = ""
        if "APPDATA" in os.environ and sys.platform.startswith("win"):
            versiondir = os.path.join(os.environ['APPDATA'], ".minecraft", "versions")
        elif "HOME" in os.environ:
            # For linux:
            versiondir = os.path.join(os.environ['HOME'], ".minecraft", "versions")
            if not os.path.exists(versiondir) and sys.platform.startswith("darwin"):
                # For Mac:
                versiondir = os.path.join(os.environ['HOME'], "Library",
                                          "Application Support", "minecraft", "versions")

        try:
            if verbose: logging.info("Looking in the following directory: \"%s\"" % versiondir)
            versions = os.listdir(versiondir)
            if verbose: logging.info("Found these versions: {0}".format(versions))
        except OSError:
            # Directory doesn't exist? Ignore it. It will find no versions and
            # fall through the checks below to the error at the bottom of the
            # method.
            versions = []

        available_versions = []
        for version in versions:
            # Look for the latest non-snapshot that is at least 1.8. This
            # version is only compatible with >=1.8, and we cannot in general
            # tell if a snapshot is more or less recent than a release.

            # Allow two component names such as "1.8" and three component names
            # such as "1.8.1"

            if version.count(".") not in (1,2):
                continue
            try:
                versionparts = [int(x) for x in version.split(".")]
            except ValueError:
                continue

            if versionparts < [1,8]:
                continue

            available_versions.append(versionparts)
        logger.debug(available_versions)

        available_versions.sort(reverse=True)
        if not available_versions:
            if verbose: logging.info("Did not find any non-snapshot minecraft jars >=1.8.0")
        while(available_versions):
            most_recent_version = available_versions.pop(0)
            if verbose: logging.info("Trying {0}. Searching it for the file...".format(".".join(str(x) for x in most_recent_version)))

            jarname = ".".join(str(x) for x in most_recent_version)
            jarpath = os.path.join(versiondir, jarname, jarname + ".jar")

            if os.path.isfile(jarpath):
                jar = zipfile.ZipFile(jarpath)
                try:
                    jar.getinfo(filename)
                    if verbose: logging.info("Found %s in '%s'", filename, jarpath)
                    self.jars[jarpath] = jar
                    return jar.open(filename)
                except (KeyError, IOError) as e:
                    pass

            if verbose: logging.info("Did not find file {0} in jar {1}".format(filename, jarpath))

        # Last ditch effort: look for the file is stored in with the overviewer
        # installation. We include a few files that aren't included with Minecraft
        # textures. This used to be for things such as water and lava, since
        # they were generated by the game and not stored as images. Nowdays I
        # believe that's not true, but we still have a few files distributed
        # with overviewer.
        if verbose: logging.info("Looking for texture in overviewer_core/data/textures")

        logging.debug(filename)
        path = os.path.join(programdir, "overviewer_core", "data", filename)
        if os.path.isfile(path):
            if verbose: logging.info("Found %s in '%s'", filename, path)
            return open(path, mode)
        elif hasattr(sys, "frozen") or imp.is_frozen("__main__"):
            # windows special case, when the package dir doesn't exist
            path = os.path.join(programdir, "textures", filename)
            if os.path.isfile(path):
                if verbose: logging.info("Found %s in '%s'", filename, path)
                return open(path, mode)

        raise AssetLoaderException("Could not find the textures while searching for '{0}'. Try specifying the 'texturepath' option in your config file.\nSet it to the path to a Minecraft Resource pack.\nAlternately, install the Minecraft client (which includes textures)\nAlso see <http://docs.overviewer.org/en/latest/running/#installing-the-textures>\n(Remember, this version of Overviewer requires a 1.15-compatible resource pack)\n(Also note that I won't automatically use snapshots; you'll have to use the texturepath option to use a snapshot jar)".format(filename))

    @lru_cache()
    def load_image_texture(self, filename)->Image:
        # Textures may be animated or in a different resolution than 16x16.
        # This method will always return a 16x16 image

        img = self.load_img(filename)

        w,h = img.size
        if w != h:
            img = img.crop((0,0,w,w))
        if w != 16:
            img = img.resize((16, 16), Image.ANTIALIAS)

        # self.texture_cache[filename] = img
        return img


    @lru_cache()
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

    @lru_cache()
    def load_json(self, name: str, directory: str) -> dict:
        # fp = self.textures.find_file("%s/%s.json" % (directory, name), "r")
        logger.debug(directory)
        logger.debug(name)
        # with...
        with self.load_file(directory, name, ".json") as f:
            return json.load(f)
        # finally:
        #     fp.close()
        # return data

    def load_blockstates(self, name: str) -> dict:
        return self.load_json(name, self.BLOCKSTATES_DIR)

    def load_model(self, name: str) -> dict:
        logger.debug(name)
        return self.load_json(name, self.MODELS_DIR)
        # if ":" in name:
        #     return self.load_json(name[name.find(":")+1:], self.MODELS_DIR)
        # else:
        #     return self.load_json(name, self.MODELS_DIR)

    @staticmethod
    def combine_textures_fields(textures_field: dict, parent_textures_field: dict) -> dict:
        return {
            **{
                key: textures_field.get(value[1:], value) if value[0] == '#' else value
                for key, value in parent_textures_field.items()
            },
            **textures_field
        }
