import warnings
from random import random
from DeepNEAT.DeepNEAT_implementation.Attributes.attributes import FloatAttribute, BoolAttribute, StringAttribute

class BaseGene(object):

    def __init__(self, key):
        self.key = key

    def __str__(self):
        if self.key==0:
            return ""
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = ['{0}={1}'.format(a, getattr(self, a)) for a in attrib]
        return '{0}({1})'.format(self.__class__.__name__, ", ".join(attrib))

    def __lt__(self, other):
        assert isinstance(self.key,type(other.key)), "Cannot compare keys {0!r} and {1!r}".format(self.key,other.key)
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                "Class '{!s}' {!r} needs '_gene_attributes' not '__gene_attributes__'".format(
                    cls.__name__,cls),
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):
        assert self.key == gene2.key

        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene

class DefaultNodeGene(BaseGene):
    _gene_attributes = [StringAttribute('type_of_layer', options='conv2d dense'),
                        StringAttribute('num_of_nodes', options='512 1024 2048'),
                        StringAttribute('activation', options='relu sigmoid'),
                        StringAttribute('num_filters_conv', options='8 16 32 64'),
                        StringAttribute('kernel_size_conv', options='1 3 5 7'),
                        StringAttribute('stride_conv', options='1 2'),
                        StringAttribute('stride_pool', options='1 2'),
                        StringAttribute('poolsize_pool', options='2 3')]

    def __init__(self, key):
        assert isinstance(key, int), "DefaultNodeGene key must be an int, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):

        factors={
            "type_of_layer":10,
            "num_of_nodes":8,
            "activation":1,
            "num_filters_conv":1,
            "kernel_size_conv":2,
            "stride_conv":1.5,
            "stride_pool":1.5,
            "poolsize_pool":2
        }

        d=0.0
        if self.type_of_layer != other.type_of_layer:
            d+=factors["type_of_layer"]
        else:
            if self.type_of_layer == "dense":
                d+=(abs(float(self.num_of_nodes)-float(other.num_of_nodes))/1536)*factors["num_of_nodes"]
            else:
                if self.num_filters_conv != other.num_filters_conv:
                    d+=factors["num_filters_conv"]

                if self.kernel_size_conv != other.kernel_size_conv:
                    d+=factors["kernel_size_conv"]

                if self.stride_conv != other.stride_conv:
                    d+=factors["stride_conv"]

                if self.stride_pool != other.stride_pool:
                    d+=factors["stride_pool"]

                if self.stride_pool != other.stride_pool:
                    d+=factors["stride_pool"]

        if self.activation != other.activation:
            d+=factors["activation"]

        return d * config.compatibility_weight_coefficient

class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient