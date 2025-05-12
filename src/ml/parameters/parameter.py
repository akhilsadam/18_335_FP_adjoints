class DefaultParameters:
    def __init__(self):
        pass
    def update(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        return self
    
    def set_shapes(self, shapes):
        self.shapes = shapes
        self.in_channels = shapes[0][2] # B T C H [W L]
        return self