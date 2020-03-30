import imp
import json
import logging
import os
import re
import sys
import zipfile
from collections import OrderedDict
import PIL.Image as Image
from typing.io import IO

from typing import Optional, Sequence, Dict, Tuple, Union
from overviewer_core import util
# from dogpile.cache import make_region
#
logger = logging.getLogger(__name__)
#
# region = make_region().configure(
#     'dogpile.cache.pylibmc',
#     expiration_time = 3600,
#     arguments = {
#         'url': ["127.0.0.1"],
#     }
# )


class AssetLoaderException(Exception):
    "To be thrown when a texture is not found."
    pass

class zipFileWrapper(zipfile.ZipFile):
    def open(self, name, mode="r", pwd=None, force_zip64=False, encoding=None):
        return zipfile.ZipFile.open(self, name, pwd=pwd, force_zip64=force_zip64)
    @property
    def namelist(self):
        if not hasattr(self, "_nameList"):
            self._nameList = zipfile.ZipFile.namelist(self)
        return self._nameList

class Directory(object):
    def __init__(self, path):
        self.path = path

    @property
    def namelist(self):
        if not hasattr(self, "_nameList"):
            logger.debug("initializing namelist")
            file_set =set()

            for dir_, _, files in os.walk(self.path):
                for file_name in files:
                    rel_dir = os.path.relpath(dir_, self.path)
                    rel_file = os.path.join(rel_dir, file_name)
                    file_set.add(rel_file.replace("\\", "/"))
            self._nameList = file_set
        return self._nameList

    def open(self, name, mode="r"):
        return open(os.path.join(self.path, name), mode=mode)

    def close(self):
        pass


def opencontainer(path):
    if zipfile.is_zipfile(path):
        return zipFileWrapper(path)
    if os.path.isdir(path):
        return Directory(path)
    else:raise ValueError("Path '{0}' does not exist".format(path))

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
    """


    Todo: for some reason 1.13.2.jar breaks the zipfile module...  chaos ensues

    """

    BLOCKSTATES_DIR = "assets/minecraft/blockstates"
    MODELS_DIR = "assets/minecraft/models"
    TEXTURES_DIR = "assets/minecraft/textures"
    r_name_from_path= re.compile("^assets[\\/]([^\\/]+)[\\/]([^\\/]+)[\\/](.*?)(?:\.[^.\\/]*)?$")
    r_blockstates= re.compile("assets[\\/]([^\\/]+)[\\/](blockstates)[\\/](.*?)(?:\.[^.\\/]*)?$")
    r_textures= re.compile("assets[\\/]([^\\/]+)[\\/](textures)[\\/](.*?)(?:\.[^.\\/]*)?\.png$")
    r_models= re.compile("assets[\\/]([^\\/]+)[\\/](models)[\\/](.*?)(?:\.[^.\\/]*)?$")
    def __init__(self, texturepath, tex_size):
        self.texturepath = texturepath
        self.tex_size = tex_size
        if type(texturepath) == list:
            self.paths = texturepath
        else:
            self.paths=[texturepath]

        self.paths += self._get_default_locations()
        _path_obj = OrderedDict([(i,opencontainer(i)) for i in self.paths])
        self._blockstates_paths, self._textures_paths, self._model_paths = self.load_all_asset_paths(_path_obj)
        self.load_all_models(_path_obj)
        self.load_all_textures(_path_obj)
        self.load_all_blockstates(_path_obj)
        self._blocklist = self._get_blocklist(paths=_path_obj)

        for k,v in _path_obj.items():#close path objects
            print(type(v))
            v.close()

        pass
        # self._blockstates= self.load_blockstates()


        # self.load_all_blockstates()

    # def __getstate__(self):
    #     _dict =self.__dict__
    #     pprint(_dict)
    #     return _dict
    #
    # def __setstate__(self, state:dict):
    #     for k,v in state.items():
    #         setattr(self,k,v)

    def load_all_blockstates(self, paths):
        self._blockstates = {n: self._load_json(f, p, paths=paths) for n, (f,p) in self._blockstates_paths.items()}

    def load_all_textures(self, paths):
        # pprint(self._textures_paths)
        # pprint({n: self._load_image(f, p, paths=paths) for n, (f,p) in self._textures_paths.items()})
        self._textures = {n: self._load_image(f, p, paths=paths) for n, (f,p) in self._textures_paths.items()}

    def load_all_models(self, paths):
        # pprint(self._model_paths)
        self._models = {
            n: self.load_and_combine_model(n, p, paths=paths)[1]
                        for n, (f,p) in self._model_paths.items()
        }

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
        if "APPDATA" in os.environ and sys.platform.startswith("win"): #windows install location
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


    def load_all_asset_paths(self, paths)->Tuple[Dict[str,Tuple[str,str]], Dict[str,Tuple[str,str]], Dict[str,Tuple[str,
                                                                                                                  str]]]:
        _blockstates = {"{0}:{2}".format(*self.get_name_from_path(k)): (k,v) for k,
                                                                                     v in self.list_files_matching(
            self.r_blockstates, paths).items()}
        _textures = {"{0}:{2}".format(*self.get_name_from_path(k)): (k,v) for k, v in self.list_files_matching(
            self.r_textures, paths).items()}
        _models = {"{0}:{2}".format(*self.get_name_from_path(k)): (k,v) for k, v in self.list_files_matching(
                self.r_models, paths).items()}
        return _blockstates, _textures, _models

    def _load_file(self, file, mode="r", paths = None)->Tuple[Union[zipFileWrapper,Directory], IO]:

        # print("loading File: ", file)
        for p,c in paths.items():
            # path = opencontainer(p)
            if file in c.namelist:
                return c,c.open(file, mode=mode)

        raise ValueError("Could not find the requested file.")

    def _load_json(self, file, path=None, paths=None)->Tuple[Union[zipFileWrapper,Directory], Dict]:
        if path is None:
            with self._load_file(file, mode="r", paths=paths) as f:
                return f[0], json.load(f[1])
        else:
            with self.load_file_from_path(path=path,file=file, mode="r", paths=paths) as f:
                return path, json.load(f)


    def _load_image(self, file, path =None, paths=None, size =(16,16))->bytes:
        print("_load_image()")
        with self._load_file(file, mode="rb", paths=paths)[1]  if (path is not None ) else self.load_file_from_path(
                    path, file, mode="rb", paths=paths)[1]  as f:
            print(f)
            texture = Image.open(f)

            return  texture.copy()

    # def load_block_texture(self, texture, ext=".png", path=None)->Image:
    #     name = "assets/minecraft/textures/block/{0}{1}".format(texture, ext)
    #     # print("searching for path: ", name)
    #     if path is None:
    #         with self._load_file(name, mode="rb") as f:
    #             return Image.open(f).copy()
    #     else:
    #         with self.load_file_from_path(path,name, mode="rb") as f:
    #             return Image.open(f).copy()

    def read_items(self,type)->dict:raise NotImplemented

    def list_block_textures(self)->Dict[str,Image.Image]:

        return {n:v  for n,v in self._textures.items() if "block/" in n}

    def load_image(self, name, size:Tuple[int, int]=None, mode:str= "RGBA")->Image.Image:
        # pprint(self._textures)

        try:
            texture= self._textures[name]
        except:
            texture= self._textures[self.build_FQDN(name)]

        # texture = Image.frombytes("RGBA", (16,16), texture)
        h, w =texture.size #get images size
        if h != w:# check image is square if not (for example due to animated texture) crop shorter side
            texture = texture.crop((0,0,min(h,w),min(h,w)))
        if texture.mode != mode:
            texture = texture.convert(mode)
        if size is not None:
            texture = texture.resize(size, Image.BOX)
        texture =texture.convert("RGBA")
        return texture


    def load_blockstates(self, name)->dict:
        try:
            return self._blockstates[name]
        except:
            return self._blockstates["minecraft:{0}".format(name)]

    def _load_blockstates(self, name)->dict:

        pattern = "/blockstates/.*.json"
        # print(pattern)
        p, fn = self.get_file_matching(pattern)
        # print(p, fn)

        try:
            return self._load_json(file=fn, path=p)
        except:
            # else:
            raise ValueError("Could not resolve blockstate name")

    def get_name_from_path(self, path:str)->Sequence[str]:
        "assets/minecraft/textures/block/acacia_planks.png"
        return re.match(self.r_name_from_path, path).groups()

    def build_FQDN(self, path):
        return "{0}:{2}".format(*self.get_name_from_path(path))
    # @region.cache_on_arguments()
    def load_model(self, model,  path=None, ext=".json")->Tuple[Union[Directory, zipFileWrapper], dict]:

        if ":" in model:
            model = model.split(":")[1]
        pattern = "/models/{0}{1}".format(model, ext)

        # print(pattern)
        path, data = self.get_file_matching(pattern, path)
        # print(file_dict)
        try:
            return self._load_json(data, path=path)
        except:
            raise ValueError("Could not resolve model name")

    def get_blocklist(self)->set:
        return set(self._blocklist)

    def _get_blocklist(self, paths):
        _ret = self.list_files_matching("/blockstates/.*\.json", paths=paths)
        # logger.info("Loading blocklist..")
        # pprint(set(os.path.splitext(os.path.split(bs)[1])[0] for bs in _ret.keys()))
        return  set(os.path.splitext(os.path.split(bs)[1])[0] for bs in _ret.keys())

    def list_files_matching(self, pattern:Union[str], paths)->dict:
        _ret = {}
        r_pattern = re.compile(pattern)
        for p,c in paths.items():
            # path = opencontainer(p)
            # print("Searching: ", p)

            for fn in  c.namelist:
                if bool(re.search(r_pattern, fn.replace("\\", "/"))) & (fn not in _ret.keys()):
                    # print("adding item to dict: ", fn, p)
                    _ret[fn] = p
        return _ret

    def get_file_matching(self, pattern:str, path=None)->Tuple[Union[zipFileWrapper, Directory],str]:
        _ret = {}
        r_pattern = re.compile(pattern)
        _fn =""
        if path is None:
            for path,c in self.path_objs.items():
                # path = opencontainer(p)
                # print("Searching: ", p)
                for fn in  c.namelist:
                    if bool(re.search(r_pattern, fn.replace("\\", "/"))) & (fn not in _ret.keys()):
                        # print("adding item to dict: ", fn, p)
                        _fn = fn

        else:
            if isinstance(path, str):
                if path in self.path_objs.keys():
                    # print("path found in paths")
                    path = self.path_objs[path]
                else:
                    # print("path not in paths")
                    # print(path)
                    path = opencontainer(path)

            for fn in  path.namelist:
                if bool(re.search(r_pattern, fn.replace("\\", "/"))) & (fn not in _ret.keys()):
                    _fn =fn
                    # print("adding item to dict: ", fn, p)


        return path, _fn
        # return _ret

    def load_file_from_path(self,path, file,paths, mode="r")->Optional[IO]:

        if isinstance(path, str):
            if path in paths.keys():
                # print("path found in paths")
                path = paths[path]
            else:
                # print("path not in paths")
                # print(path)
                path = opencontainer(path)
        try:
            # print(opencontainer(path).namelist)
            return path.open(file, mode=mode)
        except:
            raise ValueError("Could not load file '{0}' from path '{1}'".format(file, path))

    # @lru_cache(maxsize=16)
    def load_and_combine_model(self, name, path=None, paths=None):
        # logger.info("Building models: {0}".format(name))
        path, data = self._load_json(*self._model_paths[name], paths=paths)
        # pprint(data)
        if "parent" in data:
            parent_data = {}
            # Load the JSON from the parent
            try:
                path, parent_data = self.load_and_combine_model(data["parent"], path, paths=paths)
            except:
                try:
                    path, parent_data = self.load_and_combine_model("minecraft:{0}".format(data["parent"]), path, paths=paths)
                except:
                    try:
                        path, parent_data = self.load_and_combine_model(data["parent"].split("/")[1], path, paths=paths)
                    except:
                        if "builtin" in data["parent"]:
                            pass
                        else:raise KeyError
            elements_field = data.get("elements", parent_data.get("elements"))
            textures_field = self.combine_textures_fields(data.get("textures", {}), parent_data.get("textures", {}))
        else:
            # pprint(data)
            elements_field = data.get("elements")
            textures_field = data.get("textures", {})

        return path, {
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

    def get_model(self, name):
        try:
            return self._models[name]
        except:
            return self._models["minecraft:{0}".format(name)]

if __name__ =="__main__":
    # print("Testing Asset loader class")
    pass
    # o = objectName("minecraft:block/stone")
    # print(o.namespace)
    # print(o.type)
    # print(o.name)
    # print(objectName("minecraft:block/stone").__str__())
    # nAL = AssetLoader("C:/Users/Lonja/AppData/Roaming/.minecraft/versions/20w12a/20w12a.jar")
    # nAL = newAssetLoader(os.path.join(util.get_program_path(),"overviewer_core", "data"))
    # print((nAL.load_json("assets/minecraft/blockstates/acacia_button.json")))
    # print((nAL.build_FQDN("assets/minecraft/textures/block/acacia_planks.png")))

    # nAL.load_block_texture("acacia_planks").show()
    #
    # nAL.load_block_texture("watercolor").show()
    # pprint(nAL.list_files_matching("/blockstates/.*\.json"))
    # pprint(nAL.get_blocklist())
    # nAL.load_model("block/acacia_log")
    # nAL.load_image("assets/minecraft/textures/block/watercolor.png",
                   # os.path.join(util.get_program_path(),"overviewer_core", "data")).show()

    _t = opencontainer("E:/downloads/mmc-stable-win32/MultiMC/libraries/com/mojang/minecraft/1.13.2/minecraft-1.13.2"
                       "-client.jar")
    # _t.testzip()
    # namelist= _t.namelist
    # for i in namelist:
    #     if "stone_slab_side.png" in i:
            # print(i)
    # print(_t.open("assets/minecraft/textures/block/stone_slab_side.png"))
    # nAL.load_image("assets/minecraft/textures/block/stone.png").show()
    # pprint(nAL.list_block_textures())




