.. _io_cross_sections:

============================================
Cross Sections Listing -- cross_sections.xml
============================================

.. _directory_element:

-----------------------
``<directory>`` Element
-----------------------

The ``<directory>`` element specifies a root directory to which the path for all
files listed in a :ref:`library_element` are given relative to. This element has
no attributes or sub-elements; the directory should be given within the text
node. For example,

.. code-block:: xml

   <directory>/opt/data/cross_sections/</directory>

.. _library_element:

---------------------
``<library>`` Element
---------------------

The ``<library>`` element indicates where an HDF5 data file is located, whether
it contains incident neutron, incident photon, thermal scattering, or windowed
multipole data, and what materials are listed within. It has the following
attributes:

  :materials:

    A space-separated list of nuclides or thermal scattering tables. For
    example,

    .. code-block:: xml

       <library materials="U234 U235 U238" />
       <library materials="c_H_in_H2O c_D_in_G2O" />

    Often, just a single nuclide or thermal scattering table is contained in a
    given file. If a material is listed in more than one ``<library>`` element
    of the same type, the first one is used and a warning is issued.

  :path:
    Path to the HDF5 file. An absolute path is used as is. A relative path is
    relative to the directory given by the :ref:`directory_element` if it is
    specified, and otherwise to the directory containing the
    ``cross_sections.xml`` file. Files written by
    :meth:`openmc.data.DataLibrary.export_to_xml` use paths relative to the
    ``cross_sections.xml`` file, so the file and the data can be moved
    together.

  :type:
    The type of data contained in the file. Accepted values are 'neutron',
    'thermal', 'photon', and 'wmp'. A library of any other type is skipped
    with a warning, so that a file listing data types added in a newer version
    of OpenMC can still be used.

.. _depletion_element:

-----------------------------
``<depletion_chain>`` Element
-----------------------------

The ``<depletion_chain>`` element indicates the location of the depletion chain file.
This file contains information describing how nuclides decay and transmute to other
nuclides through the depletion process. This element has a single attribute, ``path``,
pointing to the location of the chain file.

.. code-block:: xml

    <depletion_chain path="/opt/data/chain_endfb7.xml"/>

The structure of the depletion chain file is explained in :ref:`io_depletion_chain`.
