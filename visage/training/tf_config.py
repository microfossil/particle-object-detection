from collections import OrderedDict
from dotted.collection import DottedCollection, DottedDict, DottedList


def parse_value(s):
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        try:
            return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            return s


class TFConfig(object):
    """
    Helper class to create pipeline.config files
    """
    def __init__(self):
        self.dict_tree = []
        self.data = None

    @staticmethod
    def load(filename):
        with open(filename) as f:
            parser = TFConfig()
            lines = f.readlines()
            for line in lines:
                parser.parse_line(line)
        parser.data = DottedDict(parser.data)
        return parser

    def parse_line(self, line):
        line = line.strip("\n")
        if line == "":
            return
        if len(self.dict_tree) == 0:
            self.dict_tree.append(OrderedDict())
            self.data = self.dict_tree[0]
        current_dict = self.dict_tree[-1]
        line = line.lstrip().rstrip()
        parts = line.split(" ")
        # Blank line
        if len(parts) == 0 or parts[0].startswith("#"):
            return
        # End of dictionary
        elif len(parts) == 1:
            if parts[0] == "}":
                self.dict_tree.pop()
            else:
                raise ValueError("Line {} could not be parsed".format(parts[0]))
        # Data line
        else:
            parts[1] = " ".join(parts[1:])
            # Start of dictionary
            key = parts[0]
            if parts[1] == "{":
                if key.endswith(":"):
                    key = key[:-1]
                if key not in current_dict:
                    new_dict = dict()
                    current_dict[key] = new_dict
                    self.dict_tree.append(new_dict)
                elif isinstance(current_dict[key], list):
                    new_dict = dict()
                    current_dict[key].append(new_dict)
                    self.dict_tree.append(new_dict)
                else:
                    l = []
                    l.append(current_dict[key])
                    current_dict[key] = l
                    new_dict = dict()
                    l.append(new_dict)
                    self.dict_tree.append(new_dict)

            elif key.endswith(":"):
                key = key[:-1]
                current_dict[key] = parse_value(parts[1])

    def to_tf(self):
        str = ""
        return TFConfig.format("", self.data, str, 0)

    def set(self, key, val):
        TFConfig.find_and_set_entry(self.data, key, val)

    @staticmethod
    def find_and_set_entry(obj, key, val):
        if isinstance(obj, dict) or isinstance(obj, DottedDict):
            for k, v in obj.items():
                if k == key:
                    obj[k] = val
                    return
                else:
                    if TFConfig.find_and_set_entry(v, key, val):
                        return
        elif isinstance(obj, list) or isinstance(obj, DottedList):
            for v in obj:
                if TFConfig.find_and_set_entry(v, key, val):
                    return
        else:
            pass

    def set_num_classes(self, val):
        TFConfig.find_and_set_entry(self.data, "num_classes", val)

    @staticmethod
    def format(key, val, msg, level):
        if isinstance(val, dict) or isinstance(val, DottedDict):
            if key != "":
                msg += " " * level + key + " {\n"
                level += 2
            for k, v in val.items():
                msg += TFConfig.format(k, v, "", level)
            if key != "":
                level -= 2
                msg += " " * level + "}\n"
        elif isinstance(val, list) or isinstance(val, DottedList):
            for v in val:
                # str += " " * level + key + " {\n"
                msg += TFConfig.format(key, v, "", level)
                # str += " " * level + "}\n"
        elif isinstance(val, str):
            msg += " " * level + "{}: {}\n".format(key, val)
        elif isinstance(val, bool):
            if val == True:
                msg += " " * level + "{}: true\n".format(key)
            elif val == False:
                msg += " " * level + "{}: false\n".format(key)
        else:
            msg += " " * level + "{}: {}\n".format(key, val)
        return msg

    def save_as(self, filename):
        with open(filename, "w") as f:
            f.write(self.to_tf())
