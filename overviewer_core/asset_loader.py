import imp
import json
import logging
import os
import re
import sys
import zipfile
from pprint import pprint

import PIL.Image as Image

from typing.io import IO

from overviewer_core import util

from functools import lru_cache

from typing import Optional
logger = logging.getLogger(__name__)


class AssetLoaderException(Exception):
    "To be thrown when a texture is not found."
    pass

class zipFileWrapper(zipfile.ZipFile):
    def open(self, name, mode="r", pwd=None, force_zip64=False, encoding=None):
        zipfile.ZipFile.open(self, name, pwd=pwd, force_zip64=force_zip64)

class Directory(object):
    def __init__(self, path):
        self.path = path
    def namelist(self):
        file_set =set()
        for dir_, _, files in os.walk(self.path):
            for file_name in files:
                rel_dir = os.path.relpath(dir_, self.path)
                rel_file = os.path.join(rel_dir, file_name)
                file_set.add(rel_file.replace("\\", "/"))
        return list(file_set)
    def open(self, name, mode="r"):
        return open(os.path.join(self.path, name), mode=mode)

def opencontainer(path):
    if zipfile.is_zipfile(path):
        return zipFileWrapper(path)
    return Directory(path)

class objectName(object):

    def __init__(self, name:str, type:str="block"):

        if name.count(":")==1:
            self.namespace, name  = name.split(":")
        elif name.count(":")> 1:
            raise ValueError("Name cannot currently contain more than one ':'")
        else: self.namespace = "minecraft"


        if name.count("/") ==1:
            self.type, name  = name.split("/")
        else:
            self.type=type

        self.name = name


    def __str__(self):
        return "{0}:{1}/{2}".format(self.namespace,self.type,self.name)


class AssetLoader(object):
    BLOCKSTATES_DIR = "assets/minecraft/blockstates"
    MODELS_DIR = "assets/minecraft/models"
    TEXTURES_DIR = "assets/minecraft/textures"

    def __init__(self, texturepath):
        self.texturepath = texturepath
        if type(texturepath) == list:
            self.paths = texturepath
        else:
            self.paths=[texturepath]
        self.name_from_path= re.compile(r"^assets[\\/]([^\\/]+)[\\/]([^\\/]+)[\\/](.*?)(?:\.[^.\\/]*)?$")
        self.paths += self._get_default_locations()

        # pprint(self.paths)

    #optimize:
    def _get_default_locations(self, verbose=True)->list:
        _ret = []
        _ret.append(os.path.join(util.get_program_path(),"overviewer_core", "data"))
        # Look in the location of the overviewer executable for the given path

        if sys.platform.startswith("darwin"):
            _ret.append("/Applications/Minecraft")
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
            # if verbose: logging.info("Looking in the following directory: /"%s/"" % versiondir)
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
            _ret.append(jarpath)

            # if verbose: logging.info("Did not find file {0} in jar {1}".format(filename, jarpath))
        return _ret


    def load_file(self, file, mode="r")->Optional[IO]:

        # print("loading File: ", file)
        for p in self.paths:
            path = opencontainer(p)
            if file in path.namelist():
                return path.open(file, mode=mode)
        raise ValueError("Could not find the requested file.")

    def load_json(self, file, path=None)->dict:
        if path is None:
            with self.load_file(file, mode="r") as f:
                return json.load(f)
        else:
            with self.load_file_from_path(path,file, mode="r") as f:
                return json.load(f)


    def load_image(self, file, path =None)->Image:
        if path is None:
            return Image.open(self.load_file(file, mode="rb"))
        else:
            return Image.open(self.load_file_from_path(path, file, mode="rb"))


    def load_block_texture(self, texture, ext=".png", path=None)->Image:
        name = "assets/minecraft/textures/block/{0}{1}".format(texture, ext)
        # print("searching for path: ", name)
        if path is None:
            # print(self.load_file(name, mode="rb"))
            return Image.open(self.load_file(name, mode="rb"))
        else:
            return Image.open(self.load_file_from_path(path,name, mode="rb"))


    def read_items(self,type)->list:raise NotImplemented

    def list_block_textures(self)->list:
        _tex_name_dict = self.list_files_matching("assets/minecraft/textures/block/.*\.png$")
        _tex_dict = {}
        for k,v in _tex_name_dict.items():
            _tex_dict[self.build_FQDN(k)] = self.load_image(path=v, file=k)
        return _tex_dict


    def get_name_from_path(self, path:str)->str:
        "assets/minecraft/textures/block/acacia_planks.png"
        return re.match(self.name_from_path, path).groups()

    def build_FQDN(self, path):
        return "{0}:{2}".format(*self.get_name_from_path(path))

    def load_model(self, model, ext=".json")->json:
        name = "assets/minecraft/models/{0}{1}".format(model, ext)
        print(name)
        file_dict = self.list_files_matching(name)
        print(file_dict)
        if len(file_dict) ==1:
            for k, v in file_dict.items():
                return self.load_json(k, path=v)
        else:
            raise ValueError("Could not resolve model name")

    def get_blocklist(self):
        _ret = self.list_files_matching("/blockstates/.*\.json")
        return set(os.path.splitext(os.path.split(bs)[1])[0] for bs in _ret.keys())

    def list_files_matching(self, pattern:str)->dict:
        _ret = {}
        for p in self.paths:
            path = opencontainer(p)
            # print("Searching: ", p)
            for fn in  path.namelist():
                if bool(re.search(pattern, fn.replace("\\", "/"))) & (fn not in _ret.keys()):
                    # print("adding item to dict: ", fn, p)
                    _ret[fn] = p
        return _ret


    def load_file_from_path(self,path, file, mode="r")->Optional[IO]:
        try:
            return opencontainer(path).open(file, mode=mode)
        except:
            raise ValueError("Could not load file '{0}' from path '{1}'".format(file, path))


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

    @staticmethod
    def combine_textures_fields(textures_field: dict, parent_textures_field: dict) -> dict:
        return {
            **{
                key: textures_field.get(value[1:], value) if value[0] == '#' else value
                for key, value in parent_textures_field.items()
            },
            **textures_field
        }




if __name__ =="__main__":
    print("Testing Asset loader class")
    pass
    # o = objectName("minecraft:block/stone")
    # print(o.namespace)
    # print(o.type)
    # print(o.name)
    # print(objectName("minecraft:block/stone").__str__())
    nAL = AssetLoader("C:/Users/Lonja/AppData/Roaming/.minecraft/versions/20w12a/20w12a")
    # nAL = newAssetLoader(os.path.join(util.get_program_path(),"overviewer_core", "data"))
    # print((nAL.load_json("assets/minecraft/blockstates/acacia_button.json")))
    print((nAL.build_FQDN("assets/minecraft/textures/block/acacia_planks.png")))

    # nAL.load_block_texture("acacia_planks").show()
    #
    # nAL.load_block_texture("watercolor").show()
    # pprint(nAL.list_files_matching("/blockstates/.*\.json"))
    # pprint(nAL.get_blocklist())
    # nAL.load_model("block/acacia_log")
    # nAL.load_image("assets/minecraft/textures/block/watercolor.png",
                   # os.path.join(util.get_program_path(),"overviewer_core", "data")).show()

    pprint(nAL.list_block_textures())




