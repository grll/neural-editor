class Dict2Object(dict):
    def __init__(self, dict_):
        super(Dict2Object, self).__init__(dict_)
        for key in self:
            item = self[key]
            if isinstance(item, list):
                for idx, it in enumerate(item):
                    if isinstance(it, dict):
                        item[idx] = Dict2Object(it)
            elif isinstance(item, dict):
                self[key] = Dict2Object(item)

    def __getattr__(self, key):
        return self[key]
