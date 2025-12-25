Source: `NR/einsteintoolkit/repos/einsteinexamples/par/arXiv-1111.3344/bbh/BBHMedRes.par`

```ccl
CoordBase::domainsize       = "minmax"
CoordBase::spacing          = "gridspacing"

CoordBase::xmin =   0.00
CoordBase::ymin =-120.00
CoordBase::zmin =   0.00
CoordBase::xmax = 120.00
CoordBase::ymax = 120.00
CoordBase::zmax = 120.00
CoordBase::dx   =   1.50
CoordBase::dy   =   1.50
CoordBase::dz   =   1.50

CoordBase::boundary_size_x_lower        = 3
CoordBase::boundary_size_y_lower        = 3
CoordBase::boundary_size_z_lower        = 3
CoordBase::boundary_size_x_upper        = 3
CoordBase::boundary_size_y_upper        = 3
CoordBase::boundary_size_z_upper        = 3

Driver::ghost_size               = 3

ReflectionSymmetry::reflection_x = "no"
ReflectionSymmetry::reflection_y = "no"
ReflectionSymmetry::reflection_z = "yes"

ML_BSSN::my_initial_boundary_condition = "extrapolate-gammas"
ML_BSSN::my_rhs_boundary_condition     = "NewRad"
Boundary::radpower                     = 2
```

