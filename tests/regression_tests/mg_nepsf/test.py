import os

import numpy as np

import openmc
from openmc.examples import slab_mg

from tests.testing_harness import PyAPITestHarness


def create_library():
    groups = openmc.mgxs.EnergyGroups(group_edges=[0.0, 0.625, 20.0e6])
    library = openmc.MGXSLibrary(groups)

    nu = [2.50, 2.50]
    fiss = np.array([0.002817, 0.097])
    capture = [0.008708, 0.02518]
    absorption = np.add(capture, fiss)
    total = [0.33588, 0.54628]
    chi = [1., 0.]

    # P3 Legendre moments a_l whose truncated expansions are negative in
    # part of [-1, 1]
    forward = [1.0, 0.8001, 0.5199, 0.2801]
    backward = [1.0, -0.6337, 0.3098, -0.1173]
    peaked = [1.0, 0.6667, 0.2857, 0.0714]

    for name, kernels in (('mat_1', (forward, backward, peaked)),
                          ('mat_2', (peaked, forward, backward))):
        scatter = np.zeros((2, 2, 4))
        scatter[0, 0] = 0.31980 * np.array(kernels[0])
        scatter[0, 1] = 0.004555 * np.array(kernels[1])
        scatter[1, 1] = 0.424100 * np.array(kernels[2])
        data = openmc.XSdata(name, groups)
        data.order = 3
        data.set_nu_fission(np.multiply(nu, fiss))
        data.set_absorption(absorption)
        data.set_scatter_matrix(scatter)
        data.set_total(total)
        data.set_chi(chi)
        library.add_xsdata(data.convert_scatter_format('nepsf', 3))

    library.export_to_hdf5('2g.h5')


class MGXSTestHarness(PyAPITestHarness):
    def _cleanup(self):
        super()._cleanup()
        f = '2g.h5'
        if os.path.exists(f):
            os.remove(f)


def test_mg_nepsf():
    create_library()
    model = slab_mg(num_regions=2, mat_names=['nepsf', 'micro'])
    # Combine the steps of two nuclides
    model.materials[-1] = openmc.Material(name='micro', material_id=2)
    model.materials[-1].set_density('sum')
    model.materials[-1].add_nuclide('mat_1', 0.5)
    model.materials[-1].add_nuclide('mat_2', 0.5)

    harness = MGXSTestHarness('statepoint.10.h5', model)
    harness.main()
