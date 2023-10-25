from typing import Any

from numpy import ndarray, dtype, int_, float_
from numpy.polynomial._polybase import ABCPolyBase
from numpy.polynomial.polyutils import trimcoef

__all__: list[str]

hermtrim = trimcoef

def poly2herm(pol): ...
def herm2poly(c): ...

hermdomain: ndarray[Any, dtype[int_]]
hermzero: ndarray[Any, dtype[int_]]
hermone: ndarray[Any, dtype[int_]]
hermx: ndarray[Any, dtype[float_]]

def hermline(off, scl): ...
def hermfromroots(roots): ...
def hermadd(c1, c2): ...
def hermsub(c1, c2): ...
def hermmulx(c): ...
def hermmul(c1, c2): ...
def hermdiv(c1, c2): ...
def hermpow(c, pow, maxpower=...): ...
def hermder(c, m=..., scl=..., axis=...): ...
def hermint(c, m=..., k = ..., lbnd=..., scl=..., axis=...): ...
def hermval(x, c, tensor=...): ...
def hermval2d(x, y, c): ...
def hermgrid2d(x, y, c): ...
def hermval3d(x, y, z, c): ...
def hermgrid3d(x, y, z, c): ...
def hermvander(x, deg): ...
def hermvander2d(x, y, deg): ...
def hermvander3d(x, y, z, deg): ...
def hermfit(x, y, deg, rcond=..., full=..., w=...): ...
def hermcompanion(c): ...
def hermroots(c): ...
def hermgauss(deg): ...
def hermweight(x): ...

class Hermite(ABCPolyBase):
    domain: Any
    window: Any
    basis_name: Any